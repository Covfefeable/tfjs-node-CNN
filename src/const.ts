import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';

export const charMap = {
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
        arr.push(charMap[str[i].toLowerCase() as keyof typeof charMap]);
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
        return Object.keys(charMap).find(key => charMap[key as keyof typeof charMap] === item.indexOf(Math.max(...item)));
    }).join('');
    return str;
}