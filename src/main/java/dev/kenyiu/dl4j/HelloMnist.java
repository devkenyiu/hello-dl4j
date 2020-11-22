package dev.kenyiu.dl4j;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

@Slf4j
public class HelloMnist {

    private static final int HEIGHT = 28;
    private static final int WIDTH = 28;
    private static final int DIM_INPUT = WIDTH * HEIGHT;
    private static final int DIM_OUTPUT = 10;

    public static void main(String[] args) throws Exception {
        HelloMnist hm = new HelloMnist();
        hm.run();
    }

    private void run() throws Exception {
        int batchSize = 64; // 1 iteration = 1 batch of examples pass through the network (forward & backward)
        int nEpoches = 15; // 1 epoch = all batches pass through the network
        int rngSeed = 123456789;
        DataSetIterator dsiTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator dsiTest = new MnistDataSetIterator(batchSize, false, rngSeed);

        buildModel(NN_784_1000_10(), nEpoches, dsiTrain, dsiTest);
    }

    private void buildModel(
        MultiLayerConfiguration conf,
        int nEpoches,
        DataSetIterator dsiTrain,
        DataSetIterator dsiTest
    ) {
        log.info("Building model...");
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.setListeners(new ScoreIterationListener(100)); // print score per 100 iterations
        model.init();

        log.info("Training model...");
        long t0 = System.currentTimeMillis();
        model.fit(dsiTrain, nEpoches);
        log.info("Time taken for training: {} seconds", (System.currentTimeMillis() - t0) / 1000.);

        log.info("Evaluating model...");
        Evaluation eval = model.evaluate(dsiTest);
        log.info("Classification statistics: {}", eval.stats());
    }

    private MultiLayerConfiguration NN_784_1000_10() {
        int rngSeed = 0;
        int nHiddenLayerNodes = 1000;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngSeed)
            .updater(new Nesterovs(0.006, 0.9))
            .l2(1e-4)
            .list()
            .layer(
                new DenseLayer.Builder()
                    .nIn(DIM_INPUT).nOut(nHiddenLayerNodes)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER)
                    .build()
            )
            .layer(
                new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(nHiddenLayerNodes).nOut(DIM_OUTPUT)
                    .activation(Activation.SOFTMAX)
                    .weightInit(WeightInit.XAVIER)
                    .build()
            )
            .build();

        return conf;
    }
}
