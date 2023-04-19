import Express from 'express'
import cookieParser from 'cookie-parser';
import bodyParser from 'body-parser';

import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';

import train from './src/train';
let app = Express();
let PORT = 3000;

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));
app.use(cookieParser());

app.get('/', async (req, res) => {
    const a = tf.oneHot(tf.tensor([[1, 2, 3, 4], [2, 3, 4, 5]]).cast('int32'), 36).reshape([2, 4, 36]);
    console.log(a, a.shape)
    res.send(a.dataSync());
});

app.get('/train', async (req, res) => {
    const r = await train()
    // @ts-ignore
    res.send(r);
});

app.set('port', process.env.PORT || 3000);
let server = app.listen(app.get('port'), function () {
    console.log('Express server listening...');
});