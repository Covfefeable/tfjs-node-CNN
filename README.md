## Captcha recognition using @tensorFlow/tfjs-node

to run this project you need to have prepared at least 10k captcha images that used to train. if you don't have the dataset, just unzip the `images.zip`  into the `src` folder.

### Usage

before training the model, take a look at the `config` variable in `src/train.ts` and adjust the value to your needs. 

```bash
# install dependencies
pnpm install

# start the server
pnpm dev

# train the model
curl http://localhost:3000/train

# predict the captcha
curl http://localhost:3000/predict?url={captcha-image-url}
```