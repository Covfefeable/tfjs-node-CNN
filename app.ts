import Express from 'express'
import cookieParser from 'cookie-parser';
import bodyParser from 'body-parser';

import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';

import train from './src/train';
import { imgToTensor, labelToStr } from './src/const';
import { getRandcode } from './src/get-image';
let app = Express();
let PORT = 3000;

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));
app.use(cookieParser());

app.get('/', async (req, res) => {
    // 加载./model里面的模型
    const model = await tf.loadLayersModel('file://./model/model.json');
    // 预测
    const testImage = imgToTensor('./images/validation/test.jpg');
    const a = await model.predict(testImage) as tf.Tensor;
    res.send(labelToStr(Array.from(a.dataSync())));
});

app.get('/train', async (req, res) => {
    const r = await train()
    // @ts-ignore
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