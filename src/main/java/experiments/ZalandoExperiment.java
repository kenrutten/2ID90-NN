package experiments;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.activation.RELU;
import nl.tue.s2id90.dl.NN.initializer.Gaussian;
import nl.tue.s2id90.dl.NN.layer.Flatten;
import nl.tue.s2id90.dl.NN.layer.FullyConnected;
import nl.tue.s2id90.dl.NN.layer.InputLayer;
import nl.tue.s2id90.dl.NN.layer.OutputSoftmax;
import nl.tue.s2id90.dl.NN.layer.SimpleOutput;
import nl.tue.s2id90.dl.NN.loss.CrossEntropy;
import nl.tue.s2id90.dl.NN.loss.MSE;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.optimizer.SGD;
import nl.tue.s2id90.dl.NN.optimizer.update.MyGradientDescentVariant;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.NN.transform.MeanSubtraction;
import nl.tue.s2id90.dl.NN.validate.Classification;
import nl.tue.s2id90.dl.experiment.Experiment;
import nl.tue.s2id90.dl.experiment.GUIExperiment;
import nl.tue.s2id90.dl.input.InputReader;
import nl.tue.s2id90.dl.input.MNISTReader;
import nl.tue.s2id90.dl.javafx.FXGUI;
import nl.tue.s2id90.dl.javafx.ShowCase;


public class ZalandoExperiment extends GUIExperiment {
    int batchSize = 32;
    int epochs = 5; //# of epochs that a training takes
    double learningRate = 0.02;
    int m = 784;
    int n = 10;
    String[] labels = {
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        };
    ShowCase showCase = new ShowCase(i -> labels[i]);
	// (hyper)parameteres
	// ...

	public void go() throws IOException {
		// you are going to add code here
        // read input and print some information on the data
        InputReader reader = MNISTReader.fashion(batchSize);
        System.out.println("Reader info:\n" + reader.toString());
        
        FXGUI.getSingleton().addTab("show case", showCase.getNode());
        showCase.setItems(reader.getValidationData(100));
        
        //Use meansubstraction
        /*MeanSubtraction ms = new MeanSubtraction();
        ms.fit(reader.getTrainingData()); //retrieve mean
        ms.transform(reader.getTrainingData()); //apply transformation
        ms.transform(reader.getValidationData()); //apply transformation*/

        // print a record
        reader.getValidationData(1).forEach(System.out::println);
        
        Model model = createModel();
        System.out.println(model);
        
        Optimizer sgd = SGD.builder()
                .model(model)
                .learningRate(learningRate)
                .validator(new Classification())
                .updateFunction(MyGradientDescentVariant::new)
                .build();
        trainModel(sgd, reader, epochs, 0);
	}

	public static void main(String[] args) throws IOException {
		new ZalandoExperiment().go();
	}
    
    Model createModel() {
        Model model = new Model(new InputLayer("In", new TensorShape(28, 28, 1), true));
        model.addLayer(new Flatten("Flatten", new TensorShape(28, 28, 1)));
        model.addLayer(new OutputSoftmax("Out",
                new TensorShape(m), n, new CrossEntropy()));
        
        model.initialize(new Gaussian());
        return model;
    }
    
    public void onEpochFinished(Optimizer sgd, int epoch){
        super.onEpochFinished(sgd, epoch);
        showCase.update(sgd.getModel());
    }
    
}