// 从 http://xxx.com/api/v1/image/getRandcode 获取验证码图片，识别并保存到本地的 images/new-train 文件夹中作为训练样本

import fs from 'fs';
import path from 'path';
import axios from 'axios';

const IMAGEPATH = './images/new-train';

export const getRandcode = async () => {
    const { data } = await axios.get('https://xxx.com/api/v1/image/getRandcode', {
        responseType: 'arraybuffer'
    });
    const base64 = Buffer.from(data, 'binary').toString('base64');
    const image = base64.replace(/^data:image\/\w+;base64,/, '');
    const imageBuffer = Buffer.from(image, 'base64');

    // 请求 打码平台 识别验证码，返回识别结果

    const res = await axios.post('https://xxx.xxx.net/Upload/Processing.php', {
        file_base64: base64
    });

    if (res.data.pic_str && res.data.pic_str.length === 4) {
        const fileName = `${res.data.pic_str}.jpg`;
        fs.writeFileSync(path.join(IMAGEPATH, fileName), imageBuffer);
    }
}