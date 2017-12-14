package sortomania;

import java.util.Arrays;
import java.util.Comparator;

public class DataSets {

	private int[][] task1DataSets;//random
	private int[][] task1DataSetsSorted;
	private double[] task1Median;

	private String[][] task2DataSets;//random
	private String[][] task2DataSetsSorted;//random
	private String[] task2Key;
	private int[] task2Index;

	private int[][] task3DataSets;//mostly sorted
	private int[][] task3DataSetsSorted;//mostly sorted
	private double[] task3Median;

	private int[][][] task4DataSets;//2D arrays
	private int[][][] task4DataSetsSorted;//2D arrays
	private double[] task4Median;

	private Generic[][] task5DataSets;//2D arrays
	private Generic[][] task5DataSetsSorted;//2D arrays
	private int[] task5Index;
	private Generic[] task5Key;

	public static final int startingReps = 50;
	private int reps = 50;
	public static final int tasks = 5;


	public DataSets(int multiplier) {
		reps = reps * multiplier; 
		int l = 10000;

		Runner.screen.displayMessage("Preparing data for test 1...");
		task1DataSets = new int[reps][];
		task1DataSetsSorted = new int[reps][];
		task1Median = new double[reps];
		for(int i = 0; i < task1DataSets.length; i++){
			l = 10000+(int)(Math.random()*1000);
			task1DataSets[i] = randomIntsArr(l);
			int[] copy = new int[l];
			for(int j = 0; j < copy.length; j++){
				copy[j] = task1DataSets[i][j];
			}
			Arrays.sort(copy);
			task1DataSetsSorted[i] = copy; 
			task1Median[i] = median(copy);
		}

		Runner.screen.displayMessage("Preparing data for test 2...");
		task2DataSets = new String[reps][];
		task2DataSetsSorted = new String[reps][];
		task2Key = new String[reps];
		task2Index = new int[reps];
		for(int i = 0; i < task2DataSets.length; i++){
			l = 10000+(int)(Math.random()*1000);
			task2DataSets[i] = randomStringArr(l, 5);
			String[] copy = new String[l];
			for(int j = 0; j < copy.length; j++){
				copy[j] = task2DataSets[i][j];
			}
			Arrays.sort(copy);
			task2DataSetsSorted[i] = copy; 
			task2Index[i] =(int)(Math.random() * task2DataSets.length);
			task2Key[i] = copy[task2Index[i]];
		}


		Runner.screen.displayMessage("Preparing data for test 3...");
		task3DataSets = new int[reps][];
		task3DataSetsSorted = new int[reps][];
		task3Median = new double[reps];
		for(int i = 0; i < task3DataSets.length; i++){
			l = 10000+(int)(Math.random()*1000);
			task3DataSets[i] = randomIntsArr(l);
			//this one is mostly sorted, so sort it first before copying
			Arrays.sort(task3DataSets[i]);
			int[] copy = new int[l];
			for(int j = 0; j < copy.length; j++){
				copy[j] = task3DataSets[i][j];
			}
			task3DataSetsSorted[i] = copy;
			task3Median[i] = median(copy);
			unSort(task3DataSets[i]);
		}

		Runner.screen.displayMessage("Preparing data for test 4...");
		task4DataSets = new int[reps][][];
		task4DataSetsSorted  = new int[reps][][];//2D arrays
		task4Median = new double[reps];
		int lw = 500;
		for(int i = 0; i < task4DataSets.length; i++){
			lw = 500+(int)(Math.random() * 50);
			task4DataSets[i] = randomIntsArr1(lw);
			int[][] copy = new int[lw][lw];
			double[] medians = new double[lw];

			for(int row = 0; row < copy.length; row++){
				int[] rowCopy = new int[lw];
				for(int col = 0; col < rowCopy.length; col++){
					rowCopy[col] = task4DataSets[i][row][col];
				}
				Arrays.sort(rowCopy);
				copy[row] = rowCopy;
				medians[row] = median(rowCopy);
			}
			Arrays.sort(medians);
			task4Median[i] = median(medians);
			task4DataSetsSorted[i] = copy;
			
		}

		Runner.screen.displayMessage("Preparing data for test 5...");
		task5DataSets = new Generic[reps][];
		task5DataSetsSorted = new Generic[reps][];
		task5Index = new int[reps];
		task5Key = new Generic[reps];
		for(int i = 0; i < task5DataSets.length; i++){
			l = 10000+(int)(Math.random() * 1000);
			task5DataSets[i] = randomGeneric(l);
			Generic[] copy = new Generic[l];
			for(int j = 0; j < copy.length; j++){
				copy[j] = task5DataSets[i][j];
			}
			Arrays.sort(copy);
			task5DataSetsSorted[i] = copy;
			task5Index[i] = (int)(Math.random()*copy.length);
			task5Key[i] = copy[task5Index[i]];
		}
	}


	private static  double median(double[] sorted) {
		if (sorted.length % 2 == 0){
			return (sorted[sorted.length/2-1] + sorted[sorted.length/2])/2.0;
		}
		return sorted[sorted.length/2];
	}

	public int getTotalTrials(){
		return reps;

	}

	public int[] getTask1Data(int trial){
		return task1DataSets[trial];
	}

	public String[] getTask2Data(int trial){
		return task2DataSets[trial];
	}

	public String getTask2Key(int trial){
		return task2Key[trial];
	}

	public int[] getTask3Data(int trial){
		return task3DataSets[trial];
	}

	public int[][] getTask4Data(int trial){
		return task4DataSets[trial];
	}

	public Generic[] getTask5Data(int trial){
		return task5DataSets[trial];
	}

	public Generic getTask5Key(int trial){
		return task5Key[trial];
	}

	public static int[] randomIntsArr(int num)
	{
		int [] arr = new int [num];
		while (num > 0)
		{
			num--;
			arr[num] = (int)(Math.random()*1000);
		}
		return arr;
	}

	public  int[][] randomIntsArr1(int num)
	{
		int [][] arr = new int[num][num];
		int i = 0;
		while (i < num)
		{
			arr[i] = randomIntsArr(num);
			i++;
		}
		return arr;
	}

	//removes any time that is 2 standard deviations from mean
	//	
	//	public double score(double[] times)
	//	{
	//		//Calculate standard deviation
	//		double sum = 0;
	//		double[] squareTimes = Arrays.copyOf(times, times.length);
	//
	//		double mean = average(times);
	//		
	//		sum = 0;
	//		for (int i=0; i<times.length; i++)
	//		{
	//
	//			sum += Math.pow((times[i] - mean),2.0); 
	//			
	//		}
	//
	//		double stdDev = Math.sqrt(sum/times.length);
	//		
	//		//remove outliers
	//		for (int i = 0;i < times.length; i++)
	//		{
	//			if (times[i] > mean + (2*stdDev))
	//			{
	//				times[i] = mean;
	//			}
	//			if (times[i] < mean - (2*stdDev))
	//			{
	//				times[i] = mean;
	//			}
	//		}
	//		sum  = average(times);
	//		System.out.println("\nAverage Time: " + sum +" milliseconds");
	//		return sum;
	//
	//	}

	public static double average(double[] times)
	{
		double sum = 0;
		for (double x : times)
		{
			sum += x;
		}
		return sum/times.length;
	}

	public static String[] randomStringArr(int num, int length)
	{

		String [] arr = new String [num];
		while (num > 0)
		{
			int i = 0;
			String s = "";
			while (i < length)
			{
				char c = (char)((Math.random()*26)+97);
				s = s + c;
				i++;
			}
			num--;
			arr[num] = s;
		}
		return arr;
	}

	public boolean checkTest1(Contestant c, int studentResponse[], double studentMedian, int trial)
	{
		int[] origArr = task1DataSetsSorted[trial];

		boolean b = Arrays.equals(studentResponse,origArr);
		boolean d = task1Median[trial] == studentMedian;
		c.successfulSort(b, trial);
		c.successfulFind(d, trial);

		return b && d;

	}
	public boolean checkTest2(Contestant c, String[] studentResponse, int studentIndex, int trial)
	{
		boolean b = Arrays.equals(studentResponse,task2DataSetsSorted[trial]);
		boolean d = studentIndex == task2Index[trial];
		c.successfulSort(b, trial);
		c.successfulFind(d, trial);

		return b && d;
	}

	public Generic[] randomGeneric(int x)
	{
		Generic[] arr = new Generic[x];

		for (int i = 0; i < x; i++)
		{
			arr[i] = new Generic();
		}
		return arr;
	}

	public boolean checkTest5(Contestant c,Generic[] studentResponse, int studentIndex, int trial)
	{
		if(studentResponse.length != task5DataSetsSorted[trial].length){
			c.successfulSort(false, trial);
			c.successfulFind(false, trial);
			return false;
		}
		boolean b = true;
		for(int i = 0; i < studentResponse.length; i++){
			if(studentResponse[i].getValue() != task5DataSetsSorted[trial][i].getValue()){
				b = false;
			}
		}
		boolean d = studentIndex == task5Index[trial];
		c.successfulSort(b, trial);
		c.successfulFind(d, trial);

		return b && d;

	}
	public boolean checkTest3(Contestant c, int studentResponse[], double studentMedian, int trial)
	{
		int[] origArr = task3DataSetsSorted[trial];
		boolean b = Arrays.equals(studentResponse,origArr);
		boolean d = task3Median[trial] == studentMedian;
		c.successfulSort(b, trial);
		c.successfulFind(d, trial);

		return b && d;

	}
    public static boolean equals(int[] a, int[] a2) {
        if (a==a2)
            return true;
        if (a==null || a2==null)
            return false;

        int length = a.length;
        if (a2.length != length)
            return false;

        for (int i=0; i<length; i++){
            if (a[i] != a2[i]){
            	System.out.println("At index "+i+" found mismatched elements "+a[i]+" and "+a2[i]);
                return false;
            }
    }

        return true;
    }
	
    public boolean verifiedSorted(double[] arr){
    	for(int i = 0; i < arr.length-1; i++){
    		if (arr[i] > arr[i+1])return false;
    	}
    	return true;
    }
    
    public boolean verifiedSorted(int[] arr){
    	for(int i = 0; i < arr.length-1; i++){
    		if (arr[i] > arr[i+1])return false;
    	}
    	return true;
    }
    
	public boolean checkTest4(Contestant c, int studentResponse[][], double studentMedian, int trial)
	{
		int[][] origArr = task4DataSetsSorted[trial];

		
		
		boolean b = origArr.length == studentResponse.length;
		if (b){
			for(int[] arr: studentResponse){
				if(!verifiedSorted(arr)) {
					b = false;
					break;
				}
			}
		}
		boolean d = task4Median[trial] == studentMedian;
//		for(int index = 0; index < origArr.length && origArr.length == studentResponse.length; index++){
//			if(!equals(studentResponse[index],origArr[index])){
//				b = false;
//				if(studentResponse.length != origArr.length){
//					System.out.println("The two arrays that do not match up have different lengths.");
//				}
////				System.out.println(Arrays.toString(studentResponse[index]));
////				System.out.println(Arrays.toString(origArr[index]));
//			}
//		}
//		System.out.println("arrays match = "+b);
//		if(!d){
//			for(int index = 0; index < origArr.length && origArr.length == studentResponse.length; index++){
//
//				System.out.println(Arrays.toString(studentResponse[index]));
//
//
//			}
//			System.out.println("The teacher sorted arrays:");
//			for(int index = 0; index < origArr.length && origArr.length == studentResponse.length; index++){
//
//
//
//				System.out.println(Arrays.toString(origArr[index]));
//
//			}
//			
//		}
//		System.out.println("student said median was = "+studentMedian+", while actual median is "+task4Median[trial]+" for trial "+trial);
		c.successfulSort(b, trial);
		c.successfulFind(d, trial);

		return b && d;

	}
	public void unSort(int[]arr)
	{
		int percent = (int)(Math.random()*10);
		int count = (arr.length*percent)/100;

		while (count > 0)
		{
			int spot = (int)(Math.random()*arr.length);
			int swap = (int)(Math.random()*arr.length);
			int i = arr[spot];
			arr[spot] = arr[swap];
			arr[swap] = i;
			count--;
		}

	}
	public double median (int [] arr)
	{
		if (arr.length % 2 == 1) return arr[(arr.length/2)];

		return ((arr[(arr.length/2)-1] + arr[(arr.length/2)])/2.0);
	}

}
