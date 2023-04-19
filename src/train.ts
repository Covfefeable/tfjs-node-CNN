import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';
import path from 'path';
import { MODELPATH, STRMAP, imgToTensor, labelToStr, strToLabel } from './const';
import axios from 'axios';

const IMAGEWIDTH = 80;
const IMAGEHEIGHT = 34;

const train = async () => {
    // 导入 ../images/new-train 内的所有验证码
    const validationImages = fs.readdirSync('./images/new-train');
    // labels 用于存储验证码的标签
    const labels: number[][] = [];
    const validationImagesPath = validationImages.map((image) => {
        const label = image.split('.')[0]
        labels.push(strToLabel(label));
        return `./images/new-train/${image}`;
    });
    const xs = tf.concat(validationImagesPath.map((image) => {
        return imgToTensor(image);
    }));

    const ys = tf.oneHot(tf.tensor(labels).cast('int32'), 36).reshape([validationImages.length, 144]).cast('float32');

    console.log(xs.shape, ys.shape)

    // 创建模型
    const model = tf.sequential();
    model.add(tf.layers.conv2d({
        inputShape: [IMAGEHEIGHT, IMAGEWIDTH, 1],
        kernelSize: 3,
        filters: 8,
        strides: 1,
        activation: 'relu',
    }));
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));
    model.add(tf.layers.conv2d({
        kernelSize: 3,
        filters: 16,
        strides: 1,
        activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));
    model.add(tf.layers.conv2d({
        kernelSize: 3,
        filters: 32,
        strides: 1,
        activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({
        units: 256,
        activation: 'relu'
    }));
    model.add(tf.layers.dense({
        units: 144,
        activation: 'softmax'
    }));

    // 编译模型
    model.compile({
        optimizer: tf.train.adam(),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    console.log(model.summary());

    try {
        // 训练模型
        await model.fit(xs, ys, {
            batchSize: 32,
            epochs: 10,
            validationSplit: 0.1,
            // @ts-ignore
            onEpochEnd: async (epoch: any, logs: any) => {
                console.log('Epoch: ' + epoch + ' Loss: ' + logs.loss);
            }
        });
    } catch (error) {
        // @ts-ignore
        console.log(error);
        return JSON.stringify(error);
    }


    // 保存模型
    await model.save(MODELPATH);
    return 'success';

}

export default train;