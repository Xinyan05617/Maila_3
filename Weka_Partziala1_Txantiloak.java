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
        Instance lehenInst = data.instance(0);       // 第0行 -> Instance
        Attribute lehenAtrib = data.attribute(0);   // 第0列 -> Attribute
        Attribute azkenAtrib = data.attribute(data.numAttributes()-1); // 最后一列 -> Attribute,一般也是klasea
        int classIndex = data.classIndex();             // class列索引 -> int
        Attribute klaseAtrib = data.classAttribute(); // class列 -> Attribute, 属性名字，类型
        AttributeStats stats = data.attributeStats(i); //统计信息, min, max, missing...

        //AttributeStats 
        int instKop = stats.totalCount;
        int ?Kop = stats.missingCount;
        int ezberdinKop = stats.distinctCount;     
        int bakarraKop = stats.uniqueCount;       
        int[] atributuBalioKop = stats.nominalCounts; // nominal 每个值的频率,如果是numeric，会是null
        stats.numericStats // mean, min, max, stdDev (double) -->  numericStats 里面有 mean, min, max, stdDev，如果不是numeric，会是null
        double balioa = stats.numericStats.mean/min/max/stdDev;
          
        //Atributua
        Attribute atrib = data.attribute(0);
        string atribIzena = atrib.name();
        boolean mota = atrib.isNumeric(); / isNominal(); 

        // ==============================
        // Klase minoritarioa lortu (Klase atributuko balien artean, gutxiena)
        // ==============================


