import React, {useEffect, useState} from 'react';
import './App.css';
import * as tf from '@tensorflow/tfjs';

const IdolPage: React.FC = () => {
    const [model, setModel]: [tf.LayersModel | null, any] = useState(null);
    const [imageSize, setImageSize] = useState(1);
    const [noiseDim, setNoiseDim] = useState(1);

    let hCanvas: HTMLCanvasElement | null;
    let dCanvas: HTMLCanvasElement | null;

    useEffect(() => {
        (async () => {
            const tfjs = await tf.loadLayersModel("/tfjs/model.json");
            setModel(tfjs);
            setImageSize(tfjs.outputs[0].shape[1] || 0);
            setNoiseDim(tfjs.inputs[0].shape[1] || 0);
        })()
    }, [])


    const generate = async () => {
        if (model == null) {
            return;
        }
        const x = tf.randomNormal([1, noiseDim]);
        const prediction = await model!.predict(x) as tf.Tensor;
        const data = await prediction.data();

        const hContext = hCanvas?.getContext("2d");
        const dContext = dCanvas?.getContext("2d");

        if (hContext == null || dContext == null) {
            return;
        }

        const img = hContext.createImageData(imageSize, imageSize);
        for (let i = 0; i < imageSize * imageSize; i++) {
            img.data[i * 4] = data[i * 3] * 255;
            img.data[i * 4 + 1] = data[i * 3 + 1] * 255;
            img.data[i * 4 + 2] = data[i * 3 + 2] * 255;
            img.data[i * 4 + 3] = 255
        }
        hContext.putImageData(img, 0, 0);

        dContext.save();
        dContext.scale(
            dContext.canvas.width / hContext.canvas.width,
            dContext.canvas.height / hContext.canvas.height
        )
        dContext.drawImage(hCanvas!, 0, 0)
        dContext.restore()
    }

    return (
        <div>
            <h1>Idol generator</h1>
            <p>{model == null ? "Model loading..." : "Model loaded."}</p>
            <button onClick={generate}>Generate!</button>

            <div style={{margin: "8px"}}>
                <canvas ref={e => hCanvas = e} id="hidden-canvas" height={imageSize + 'px'} width={imageSize + 'px'}
                        style={{display: "none"}}/>
                <canvas ref={e => dCanvas = e} id="display-canvas" height="400px" width="400px"
                        style={{border: "1px solid black"}}/>
            </div>
        </div>
    );
}

export default IdolPage;
