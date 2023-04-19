import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';


const strMap = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15, 'g': 16, 'h': 17, 'i': 18, 'j': 19, 'k': 20, 'l': 21, 'm': 22, 'n': 23, 'o': 24, 'p': 25, 'q': 26, 'r': 27, 's': 28, 't': 29, 'u': 30, 'v': 31, 'w': 32, 'x': 33, 'y': 34, 'z': 35
}

const genIntLabel = (str: string) => {
    // 将字符串转换为数字数组，长度为4
    const arr: number[] = [];
    for (let i = 0; i < str.length; i++) {
        arr.push(strMap[str[i].toLowerCase() as keyof typeof strMap]);
    }
    return arr;
}

const img2x = (imgPath: string) => {
    const buffer = fs.readFileSync(imgPath)

    // 清除中间变量，节省内存
    return tf.tidy(() => {
        // 张量
        const imgTs = tf.node.decodeImage(new Uint8Array(buffer), 1);

        // 归一化到[-1, 1]之间
        return imgTs.toFloat().div(tf.scalar(255.0)).expandDims();
    })
}

const train = async () => {
    // 导入 ../images/train 内的所有验证码
    const validationImages = fs.readdirSync('./images/train');
    // labels 用于存储验证码的标签
    const labels: number[][] = [];
    const validationImagesPath = validationImages.map((image) => {
        labels.push(genIntLabel(image.split('.')[0]));
        return `./images/train/${image}`;
    });
    const xs = tf.concat(validationImagesPath.map((image) => {
        return img2x(image);
    }));

    const ys = tf.oneHot(tf.tensor(labels).cast('int32'), 36).reshape([validationImages.length, 144]);

    console.log(xs.shape, ys.shape)

    // 创建模型
    const model = tf.sequential();
    model.add(tf.layers.conv2d({
        inputShape: [34, 80, 1],
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
        loss: 'sparseCategoricalCrossentropy',
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
    await model.save('file://./model');

    // 使用模型预测
    const testImage = img2x('./images/validation/0jyQ.jpg');
    const predictOut = model.predict(testImage) as tf.Tensor;
    console.log('预测结果：', 123);
    return 'done';

}

export default train;