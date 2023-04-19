// 从 http://sso.upuphone.com:8090/esc-sso/api/v1/image/getRandcode 获取验证码图片，并保存到本地的 images/new-train 文件夹中

import fs from 'fs';
import path from 'path';
import axios from 'axios';

const IMAGEPATH = './images/new-train';

export const getRandcode = async () => {
    const { data } = await axios.get('http://sso.upuphone.com:8090/esc-sso/api/v1/image/getRandcode', {
        responseType: 'arraybuffer'
    });
    const base64 = Buffer.from(data, 'binary').toString('base64');
    const image = base64.replace(/^data:image\/\w+;base64,/, '');
    const imageBuffer = Buffer.from(image, 'base64');

    // 请求 https://upload.chaojiying.net/Upload/Processing.php 识别验证码，返回识别结果

    const res = await axios.post('https://upload.chaojiying.net/Upload/Processing.php', {
        user: 'covfefe',
        pass: '54txwzh',
        softid: '924619',
        codetype: '8001',
        file_base64: base64
    });

    if (res.data.pic_str && res.data.pic_str.length === 4) {
        const fileName = `${res.data.pic_str}.jpg`;
        fs.writeFileSync(path.join(IMAGEPATH, fileName), imageBuffer);
    }
}