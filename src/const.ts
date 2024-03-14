import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';

export const STRMAP = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15, 'g': 16, 'h': 17, 'i': 18, 'j': 19, 'k': 20, 'l': 21, 'm': 22, 'n': 23, 'o': 24, 'p': 25, 'q': 26, 'r': 27, 's': 28, 't': 29, 'u': 30, 'v': 31, 'w': 32, 'x': 33, 'y': 34, 'z': 35
}

export const imgToTensor = (imgPath: string) => {
    const buffer = fs.readFileSync(imgPath)

    // 清除中间变量，节省内存
    return tf.tidy(() => {
        // 张量
        const imgTs = tf.node.decodeImage(new Uint8Array(buffer), 1);
        // 归一化到[-1, 1]之间
        return imgTs.toFloat().div(tf.scalar(255.0)).expandDims();
    })
}

export const strToLabel = (str: string) => {
    // 将字符串转换为数字数组，长度为4
    const arr: number[] = [];
    for (let i = 0; i < str.length; i++) {
        arr.push(STRMAP[str[i].toLowerCase() as keyof typeof STRMAP]);
    }
    return arr;
}

export const labelToStr = (label: number[]) => {
    // 数组长度为144，分成36个一组，共4组
    const arr: number[][] = [];
    for (let i = 0; i < 4; i++) {
        arr.push(label.slice(i * 36, (i + 1) * 36));
    }
    // 找出每组中最大值的索引，即为预测的字符
    const str = arr.map((item) => {
        return Object.keys(STRMAP).find(key => STRMAP[key as keyof typeof STRMAP] === item.indexOf(Math.max(...item)));
    }).join('');
    return str;
}

export const MODELPATH = 'file://./model'

export const IMAGEDATASETPATH = './images/dataset/';