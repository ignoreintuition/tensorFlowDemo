/* jshint esversion: 6*/
// jshint ignore: start
import 'babel-polyfill';
import data from './assets/admissionData.json';
import * as tf from '@tensorflow/tfjs';
function run() {
  const nEpochs = 100;
  const model = tf.sequential();
  const x = [];
  const y = [];

  data.forEach(d => {
    x.push(Number(d.GRE));
    y.push(Number(d.admit));
  });
  const xs = tf.tensor2d(x, [x.length, 1]);
  const ys = tf.tensor2d(y, [y.length, 1]);
  model.add(tf.layers.dense({units: 1, inputShape: [1]}));
  model.compile({
    loss: 'meanSquaredError',
    optimizer: tf.train.sgd(0.000001),
  });
  model.fit(xs, ys, {epochs: nEpochs}).then(h => {
    model.predict(tf.tensor2d([350], [1, 1])).print();
  });
}
export {run};
