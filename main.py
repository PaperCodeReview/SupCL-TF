import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tqdm
import argparse

from common import set_seed
from common import get_logger
from common import get_session
from common import search_same
from common import create_stamp
from dataloader import set_dataset
from dataloader import dataloader
from dataloader import dataloader_supcon
from model import create_model
from loss import crossentropy
from loss import supervised_contrastive
from callback import OptionalLearningRateSchedule
from callback import create_callbacks

import tensorflow as tf


def main(args):
    args, initial_epoch = search_same(args)
    if initial_epoch == -1:
        # training was already finished!
        return

    elif initial_epoch == 0:
        # first training or training with snapshot
        args.stamp = create_stamp()

    get_session(args)
    logger = get_logger("MyLogger")
    for k, v in vars(args).items():
        logger.info("{} : {}".format(k, v))


    ##########################
    # Strategy
    ##########################
    # strategy = tf.distribute.MirroredStrategy()
    strategy = tf.distribute.experimental.CentralStorageStrategy()
    assert args.batch_size % strategy.num_replicas_in_sync == 0

    logger.info('{} : {}'.format(strategy.__class__.__name__, strategy.num_replicas_in_sync))
    logger.info("GLOBAL BATCH SIZE : {}".format(args.batch_size))
    logger.info("BATCH SIZE PER REPLICA : {}".format(args.batch_size // strategy.num_replicas_in_sync))


    ##########################
    # Dataset
    ##########################
    trainset, valset = set_dataset(args)
    steps_per_epoch = args.steps or len(trainset) // args.batch_size
    validation_steps = len(valset) // args.batch_size

    logger.info("TOTAL STEPS OF DATASET FOR TRAINING")
    logger.info("========== trainset ==========")
    logger.info("    --> {}".format(len(trainset)))
    logger.info("    --> {}".format(steps_per_epoch))

    logger.info("=========== valset ===========")
    logger.info("    --> {}".format(len(valset)))
    logger.info("    --> {}".format(validation_steps))

    ##########################
    # Model & Metric & Generator
    ##########################
    with strategy.scope():
        model = create_model(args, logger)
        if args.summary:
            model.summary()
            return

        # metrics
        metrics = {
            'loss'    :   tf.keras.metrics.Mean('loss', dtype=tf.float32),
            'val_loss':   tf.keras.metrics.Mean('val_loss', dtype=tf.float32),
        }

        # optimizer
        lr_scheduler = OptionalLearningRateSchedule(args, steps_per_epoch, initial_epoch)
        if args.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(lr_scheduler, momentum=.9, decay=.0001)
        elif args.optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(lr_scheduler)
        elif args.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr_scheduler)

        # loss
        if args.loss == 'supcon':
            criterion = supervised_contrastive(args, args.batch_size // strategy.num_replicas_in_sync)
        else:
            criterion = crossentropy(args)
            metrics['acc'] = tf.keras.metrics.CategoricalAccuracy('acc', dtype=tf.float32)
            metrics['val_acc'] = tf.keras.metrics.CategoricalAccuracy('val_acc', dtype=tf.float32)

        # generator
        if args.loss == 'crossentropy':
            train_generator = dataloader(args, trainset, 'train', args.batch_size)
            val_generator = dataloader(args, valset, 'val', args.batch_size, shuffle=False)
        elif args.loss =='supcon':
            train_generator = dataloader_supcon(args, trainset, 'train', args.batch_size)
            val_generator = dataloader_supcon(args, valset, 'train', args.batch_size, shuffle=False)
        else:
            raise ValueError()
        
        train_generator = strategy.experimental_distribute_dataset(train_generator)
        val_generator = strategy.experimental_distribute_dataset(val_generator)

    csvlogger, train_writer, val_writer = create_callbacks(args, metrics)
    logger.info("Build Model & Metrics")

    ##########################
    # READY Train
    ##########################
    train_iterator = iter(train_generator)
    val_iterator = iter(val_generator)
        
    # @tf.function
    def do_step(iterator, mode):
        def get_loss(inputs, labels, training=True):
            logits = tf.cast(model(inputs, training=training), tf.float32)
            loss = criterion(labels, logits)
            loss_mean = tf.nn.compute_average_loss(loss, global_batch_size=args.batch_size)
            return logits, loss, loss_mean

        def step_fn(from_iterator):
            if args.loss == 'supcon':
                (img1, img2), labels = from_iterator
                inputs = tf.concat([img1, img2], axis=0)
            else:
                inputs, labels = from_iterator
            
            if mode == 'train':
                with tf.GradientTape() as tape:
                    logits, loss, loss_mean = get_loss(inputs, labels)

                grads = tape.gradient(loss_mean, model.trainable_variables)
                optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
            else:
                logits, loss, loss_mean = get_loss(inputs, labels, training=False)

            if args.loss == 'crossentropy':
                metrics['acc' if mode == 'train' else 'val_acc'].update_state(labels, logits)

            return loss

        loss_per_replica = strategy.run(step_fn, args=(next(iterator),))
        loss_mean = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss_per_replica, axis=0)
        metrics['loss' if mode == 'train' else 'val_loss'].update_state(loss_mean)
        

    ##########################
    # Train
    ##########################
    for epoch in range(initial_epoch, args.epochs):
        print('\nEpoch {}/{}'.format(epoch+1, args.epochs))
        print('Learning Rate : {}'.format(optimizer.learning_rate(optimizer.iterations)))

        # train
        print('Train')
        progBar_train = tf.keras.utils.Progbar(steps_per_epoch, stateful_metrics=metrics.keys())
        for step in range(steps_per_epoch):
            do_step(train_iterator, 'train')
            progBar_train.update(step, values=[(k, v.result()) for k, v in metrics.items() if not 'val' in k])

            if args.tensorboard and args.tb_interval > 0:
                if (epoch*steps_per_epoch+step) % args.tb_interval == 0:
                    with train_writer.as_default():
                        for k, v in metrics.items():
                            if not 'val' in k:
                                tf.summary.scalar(k, v.result(), step=epoch*steps_per_epoch+step)

        if args.tensorboard and args.tb_interval == 0:
            with train_writer.as_default():
                for k, v in metrics.items():
                    if not 'val' in k:
                        tf.summary.scalar(k, v.result(), step=epoch)

        # val
        print('\n\nValidation')
        progBar_val = tf.keras.utils.Progbar(validation_steps, stateful_metrics=metrics.keys())
        for step in range(validation_steps):
            do_step(val_iterator, 'val')
            progBar_val.update(step, values=[(k, v.result()) for k, v in metrics.items() if 'val' in k])
    
        # logs
        logs = {k: v.result().numpy() for k, v in metrics.items()}
        logs['epoch'] = epoch + 1

        if args.checkpoint:
            if args.loss == 'supcon':
                ckpt_path = '{:04d}_{:.4f}.h5'.format(epoch+1, logs['val_loss'])
            else:
                ckpt_path = '{:04d}_{:.4f}_{:.4f}.h5'.format(epoch+1, logs['val_acc'], logs['val_loss'])

            model.save_weights(
                os.path.join(
                    args.result_path, 
                    '{}/{}/checkpoint'.format(args.dataset, args.stamp),
                    ckpt_path))

            print('\nSaved at {}'.format(
                os.path.join(
                    args.result_path, 
                    '{}/{}/checkpoint'.format(args.dataset, args.stamp),
                    ckpt_path)))

        if args.history:
            csvlogger = csvlogger.append(logs, ignore_index=True)
            csvlogger.to_csv(os.path.join(args.result_path, '{}/{}/history/epoch.csv'.format(args.dataset, args.stamp)), index=False)

        if args.tensorboard:
            with train_writer.as_default():
                tf.summary.scalar('loss', metrics['loss'].result(), step=epoch)
                if args.loss == 'crossentropy':
                    tf.summary.scalar('acc', metrics['acc'].result(), step=epoch)

            with val_writer.as_default():
                tf.summary.scalar('val_loss', metrics['val_loss'].result(), step=epoch)
                if args.loss == 'crossentropy':
                    tf.summary.scalar('val_acc', metrics['val_acc'].result(), step=epoch)
        
        for k, v in metrics.items():
            v.reset_states()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",       type=str,       default='resnet50')
    parser.add_argument("--batch_size",     type=int,       default=32,
                        help="batch size per replica")
    parser.add_argument("--classes",        type=int,       default=200)
    parser.add_argument("--dataset",        type=str,       default='cub')
    parser.add_argument("--img_size",       type=int,       default=224)
    parser.add_argument("--steps",          type=int,       default=0)
    parser.add_argument("--epochs",         type=int,       default=100)

    parser.add_argument("--optimizer",      type=str,       default='sgd')
    parser.add_argument("--lr",             type=float,     default=.001)
    parser.add_argument("--loss",           type=str,       default='crossentropy', choices=['crossentropy', 'supcon'])
    parser.add_argument("--temperature",    type=float,     default=0.007)

    parser.add_argument("--augment",        type=str,       default='sim')
    parser.add_argument("--randaug_layer",  type=int,       default=2)
    parser.add_argument("--standardize",    type=str,       default='minmax1',      choices=['minmax1', 'minmax2', 'norm', 'eachnorm'])

    parser.add_argument("--checkpoint",     action='store_true')
    parser.add_argument("--history",        action='store_true')
    parser.add_argument("--tensorboard",    action='store_true')
    parser.add_argument("--tb_interval",    type=int,       default=0)
    parser.add_argument("--lr_mode",        type=str,       default='constant',     choices=['constant', 'exponential', 'cosine'])
    parser.add_argument("--lr_value",       type=float,     default=.1)
    parser.add_argument("--lr_interval",    type=str,       default='20,50,80')
    parser.add_argument("--lr_warmup",      type=int,       default=0)

    parser.add_argument('--src_path',       type=str,       default='.')
    parser.add_argument('--data_path',      type=str,       default=None)
    parser.add_argument('--result_path',    type=str,       default='./result')
    parser.add_argument('--snapshot',       type=str,       default=None)
    parser.add_argument("--gpus",           type=str,       default=-1)
    parser.add_argument("--summary",        action='store_true')
    parser.add_argument("--ignore_search",  type=str,       default='')

    main(parser.parse_args())