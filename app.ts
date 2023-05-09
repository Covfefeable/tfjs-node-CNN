import Express from 'express'
import cookieParser from 'cookie-parser';
import bodyParser from 'body-parser';
import path from 'path';

import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';

import train from './src/train';
import { imgToTensor, labelToStr } from './src/const';
import { getRandcode } from './src/get-image';
import axios from 'axios';
let app = Express();
let PORT = 3000;

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));
app.use(cookieParser());

app.get('/', async (req, res) => {
    // 加载./model里面的模型
    const model = await tf.loadLayersModel('file://./model/model.json');

    // 抓取一张图片并使用模型预测
    const { data } = await axios.get('https://xxx.com/api/v1/image/getRandcode', {
        responseType: 'arraybuffer'
    });
    const base64 = Buffer.from(data, 'binary').toString('base64');
    const image = base64.replace(/^data:image\/\w+;base64,/, '');
    const imageBuffer = Buffer.from(image, 'base64');
    const fileName = `test.jpg`;
    fs.writeFileSync(path.join('./images/test', fileName), imageBuffer);

    const testImage = imgToTensor('./images/test/test.jpg');
    const a = await model.predict(testImage) as tf.Tensor;
    res.send(labelToStr(Array.from(a.dataSync())));
});

app.get('/train', async (req, res) => {
    const r = await train();
    res.send(r);
});

app.get('/generate', async (req, res) => {
    for (let i = 0; i < 500; i++) {
        console.log('remaining: ', 500 - i)
        await getRandcode()
    }
    res.send('ok');
});

app.set('port', process.env.PORT || 3000);
let server = app.listen(app.get('port'), function () {
    console.log('Express server listening...');
});