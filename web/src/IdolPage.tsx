import React, {useEffect, useState} from 'react';
import './App.css';
import * as tf from '@tensorflow/tfjs';
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';


const IdolPage: React.FC = () => {
    const [model, setModel]: [tf.LayersModel | null, any] = useState(null);
    const [imageSize, setImageSize] = useState(0);
    const [noise, setNoise]: [number[], any] = useState([]);

    let hCanvas: HTMLCanvasElement | null;
    let dCanvas: HTMLCanvasElement | null;

    useEffect(() => {
        (async () => {
            const tfjs = await tf.loadLayersModel("/tfjs/model.json");
            setModel(tfjs);
            setImageSize(tfjs.outputs[0].shape[1] || 0);
            setNoise([...Array(tfjs.inputs[0].shape[1] || 0)].map(
                (_, i) => i < 16 ? 0 : Math.random() * 8 - 4
            ));
        })()
    }, []);

    useEffect(() => {
        generate().then();
    })


    async function generate() {
        if (model == null || imageSize === 0 || noise.length === 0) {
            return;
        }
        const prediction = await model!.predict(tf.tensor2d([noise])) as tf.Tensor;
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
        dContext.scale(dContext.canvas.width / hContext.canvas.width, dContext.canvas.height / hContext.canvas.height);
        dContext.drawImage(hCanvas!, 0, 0);
        dContext.restore();
    }

    const onSliderChange = (e: number, i: number) => {
        noise[i] = e;
        generate().then();
    }

    return (
        <div>
            <h1>Idol generator</h1>
            <p>{model == null ? "Model loading..." : "Model loaded."}</p>

            <ul style={{listStyle: "none", width: "50%", margin: "auto"}}>
                {[...Array(16)].map(
                    (_, i) => {
                        return <li key={"slider-li-" + i}>
                            <Slider min={-4} max={4} step={0.5} defaultValue={0} onChange={e => onSliderChange(e, i)}/>
                        </li>
                    })}
            </ul>

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
