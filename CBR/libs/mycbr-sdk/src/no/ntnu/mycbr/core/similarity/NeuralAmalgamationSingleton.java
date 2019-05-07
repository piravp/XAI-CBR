package no.ntnu.mycbr.core.similarity;

import no.ntnu.mycbr.core.casebase.Attribute;
import no.ntnu.mycbr.core.casebase.Instance;
import no.ntnu.mycbr.core.model.AttributeDesc;
import org.datavec.api.split.StringSplit;
import org.deeplearning4j.nn.modelimport.keras.*;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class NeuralAmalgamationSingleton {
    private class MLNRunner implements Runnable {
        private MultiLayerNetwork mln;
        private INDArray inp;
        private INDArray output;
        public MLNRunner(MultiLayerNetwork mln){
            this.mln = mln;
        }
        public void setInput(INDArray inp){
            this.inp = inp;
        }
        public INDArray getOutput(){
            return this.output;

        }
        @Override
        public void run() {
            this.output = this.mln.output(this.inp);
        }
    }
    private MultiLayerNetwork mln;
    private final Log logger = LogFactory.getLog(getClass());
    private NeuralAmalgamationSingleton(String modelpath){

        //String configFile = "/home/epic/research/dataGetters/operationalmodel_1.0_cat_cross.json";
        //String weightsFile = "/home/epic/research/dataGetters/operationalmodel_1.0_cat_cross.h5";
        String configFile = modelpath+".json"; //0.88
        String weightsFile = modelpath+".h5";
        mln = null;
        try {
            mln = KerasModelImport.importKerasSequentialModelAndWeights(configFile, weightsFile);
        } catch (InvalidKerasConfigurationException e){
            logger.error("got invalid keras exception for config file"+configFile,e);
            System.out.println("invalidkerasconf");
        } catch (UnsupportedKerasConfigurationException e){
            logger.error("got unsupported keras exception for config file"+configFile,e);
            System.out.println("unsupportedkerasconf");
        } catch (IOException e ){
            logger.error("got io error on keras config file"+configFile+ " model file: "+weightsFile,e);
            System.out.println("IOexception file path is "+modelpath);
        }

    }
    private static HashMap<String,NeuralAmalgamationSingleton> instances;

    /**
     *
     * @param modelpath This is the path to the model files to be loaded into the hashmap of available models to run.
     *                  Notice this is also the key for the hashmap, so if you want to send in a new model for use, you have to provide
     *                  a new model path.
     * @return The neuralamalgamationsingleton that will give you the ability to run ANN for similarity/amalgamation
     */
    public static NeuralAmalgamationSingleton getInstance(String modelpath){
        Log staticlogger = LogFactory.getLog(NeuralAmalgamationSingleton.class);
        if(instances==null) {
            instances = new HashMap<>();
        }
        NeuralAmalgamationSingleton singleton = instances.get(modelpath);
        if(singleton!=null) {
            staticlogger.info("this model has been loaded before, fetching from hashmap");
            return singleton;
        }
        staticlogger.info("this model has NOT been loaded before, fetching from FILE");
        singleton = new NeuralAmalgamationSingleton(modelpath);
        instances.put(modelpath, singleton);
        return singleton;
    }

    public double getSolution(Instance i){
        INDArray this_inp = getArray(i);
        double this_ret = mln.output(this_inp).getDouble(0);
        return this_ret;
    }

    INDArray getArray(Instance c) {
        List<AttributeDesc> problemAttributes = new ArrayList<>();
        HashMap<String,AttributeDesc> allAttributeDesc = c.getConcept().getAllAttributeDescs();
        for(AttributeDesc attributeDesc : allAttributeDesc.values()){
            if(!attributeDesc.isSolution())
                problemAttributes.add(attributeDesc);
        }
        double[][] data = new double[1][problemAttributes.size()];
        int counter = 0;
        for(AttributeDesc attributeDesc : problemAttributes){
            Attribute att = c.getAttForDesc(attributeDesc);
            data[0][counter++] = Double.parseDouble(att.getValueAsString());
        }

        /*AttributeDesc windspeedattDesc = c.getConcept().getAllAttributeDescs().get("wind_speed");
        Attribute windspeedAttr = c.getAttForDesc(windspeedattDesc);

        AttributeDesc wind_from_direction1_desc = c.getConcept().getAllAttributeDescs().get("wind_from_direction");
        Attribute wind_from_direction_attr = c.getAttForDesc(wind_from_direction1_desc);

        AttributeDesc wind_effect_desc = c.getConcept().getAllAttributeDescs().get("wind_effect");
        Attribute wind_effect_attr = c.getAttForDesc(wind_effect_desc);

        Double windspeed = Double.parseDouble(windspeedAttr.getValueAsString());
        Double wind_from_direction = Double.parseDouble(wind_from_direction_attr.getValueAsString());
        Double windEffect = Double.parseDouble(wind_effect_attr.getValueAsString());
        double[][] arr = new double[1][3];
        arr[0][0] = windspeed;
        arr[0][1] = wind_from_direction;
        arr[0][2] = windEffect;*/
        INDArray inputarr = Nd4j.create(data);
        return inputarr;
    }
    public Similarity calculateSimilarity(Attribute value1, Attribute value2) throws Exception {
        DataSet ds = new DataSet();
        StringSplit ss = new StringSplit(value1.getValueAsString()+","+value2.getValueAsString());

        double[][] inp = new double[1][3];
        INDArray inputarr = Nd4j.create(inp);

        double[][] inp2 = new double[1][3];
        INDArray inputarr2 = Nd4j.create(inp);

        INDArray output = mln.output(inputarr);
        INDArray output2 = mln.output(inputarr2);
        return Similarity.get(output.getDouble(0)-output2.getDouble(0));
    }
    public INDArray getOutput(INDArray a){
        return mln.clone().output(a);
    }

}
