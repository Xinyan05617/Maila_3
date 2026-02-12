package azterketa;

public class azterketa{

    public static void main(String [] args) throws Exception{

        // ==============================
        // 1. Fitxategiak irakurri (BEHARREZKOA)
        // ==============================
        string sarreraFitxategia = agrg[0];

        // ==============================
        // 2. ARFF datuak kargatu (BEHARREZKOA)
        // ==============================

        DataSource source = new DataSource(sarreraFitxategia); // Datuak iturria sortu
        Instances data = source.getDataSet();          // Datu-multzoa irakurri
        data.setClassIndex(data.numAttributes() - 1);  // Weka-ri azken atributua "klasea" dela definitu
  
        // ==============================
        // *. ARFF datuak kargatu (AUKERAZKOA)
        // ==============================

        // DATA
        int instKop= data.numInstances();        // 总行数 -> int
        int atribKop = data.numAttributes();      // 总列数 -> int
        int klaseBalioKop = data.numClasses();
        Instance lehenInst = data.instance(0);       // 第0行 -> Instance
        Attribute lehenAtrib = data.attribute(0);   // 第0列 -> Attribute
        Attribute azkenAtrib = data.attribute(data.numAttributes()-1); // 最后一列 -> Attribute,一般也是klasea
        int classIndex = data.classIndex();             // class列索引 -> int
        Attribute klaseAtrib = data.classAttribute(i); // class列 -> Attribute, 属性名字，类型
        AttributeStats stats = data.attributeStats(i); //统计信息, min, max, missing...

        //AttributeStats 
        int instKop = stats.totalCount;
        int ?Kop = stats.missingCount;
        int ezberdinKop = stats.distinctCount;     //数据里面有多少个不同的值
        int bakarraKop = stats.uniqueCount;       //数据里面有多少个唯一的值
        int[] atributuBalioKop = stats.nominalCounts; // nominal 每个值的频率,如果是numeric，会是null
        stats.numericStats // mean, min, max, stdDev (double) -->  numericStats 里面有 mean, min, max, stdDev，如果不是numeric，会是null
        double balioa = stats.numericStats.mean/min/max/stdDev;
          
        //Atributua
        Attribute atrib = data.attribute(0);
        string atribIzena = atrib.name();
        int atribKop = atrib.numValues(); // 这个atributu有多少个分类
        boolean mota = atrib.isNumeric(); / isNominal(); 

        // ==============================
        // Klase minoritarioa lortu (Klase atributuko balien artean, gutxiena)
        // ==============================
        Attribute klaseaAtrib = data.classAttribute(data.classIndex());
        AttributeStats klaseaEstatistika = data.attributeStats(data.classIndex());
        int[] kotKop = klaseaEstatistika.nomialCounts;
        int maiztasunMin = kotKop[0];
        for (i=0; i < klaseaAtrib.numValues()/data.numClasses ; i++){
            String atribIzena = klaseaAtrib.value(i);
            kop atriblueKop = kotKop[i];
            if atriblueKop < maiztasunMin{
                maiztasunMin = atriblueKop;
                string klaseMin = atribIzena;
        }

        // ==============================
        // kFCV
        // ================	       
        NaiveBayes nb = new NaiveBayes();
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(nb, data, 5, new Random(1));  // crossValidateModel(eredua, datuak, fold_kopurua, random_seed)
        //crossValidateModel barruan entrenatu eta ebaluatzen du.
        
        // ==============================
        // HoldOut: Randomize + RemovePercentage / Unsupervised Resample 
        // ================	    
        Randomize randomize = new Randomize();
        randomize.setRandomSeed(1); 
        randomize.setInputFormat(data); 
        Instances dataRand = Filter.useFilter(data, randomize); // Exekutatu

        // Partiketa egin
        RemovePercentage removePer = new RemovePercentage();
        removePer.setPercentage(34.0);
        removePer.setInvertSelection(false); // Automatikoki hau da, %34 kentzen du.
        removePer.setInputFormat(dataRand);
        Instances train = Filter.useFilter(dataRand, removePer);
        removePer.setInvertSelection(true); //%66 kentzen du.
        Instances test = Filter.useFilter(dataRand, removePer);
        //Evaluatu
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(train); // Garrantzitsua: train-ekin bakarrik entrenatu!
        Evaluation eval = new Evaluation(train); // Train-en egitura behar du
        eval.evaluateModel(nb, test); // Test-ekin ebaluatu

        //RESAMPLE ERABILIZ
        Resample trainResample = new Resample();
        trainResample.setNoReplacement(true);  // Ez errepikatu
        trainResample.setSampleSizePercent(66); // %66 hartu
        trainResample.setRandomSeed(1);         // Ausazko hazia finkatu
        trainResample.setInputFormat(data);
        resample.setInvertSelection(false); // Normal: %66 hartu
        Instances train = Filter.useFilter(data, trainResample);
        resample.setInvertSelection(true); // Normal: %34 hartu
        Instances test = Filter.useFilter(data, trainResample);

        // ==============================
        // Stratified HoldOut
        // ================	   
        
        // ==============================
        // eval
        // ================	    
        eval.toMatrixString();
        eval.precision(i);
        eval.recall(i);
        eval.fMeasure(i);
        eval.weightedPrecision();
        eval.weightedRecall();
        eval.weightedFMeasure();
        

