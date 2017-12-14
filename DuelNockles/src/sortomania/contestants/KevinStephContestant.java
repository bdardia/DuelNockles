package sortomania.contestants;

import java.awt.Color;
import java.util.Arrays;

import sortomania.Contestant;

public class KevinStephContestant extends Contestant {

	public KevinStephContestant() {
		
	}
    
	public static void main(String[] args) {
		//KevinStephContestant test = new KevinStephContestant();
		//int[] arr = {4, 7, 10, 2, 18, 12, 34, 42, 23, 40, 56, 31, 8};
		//Comparable[] arr = {4, 7, 10, 2, 18, 12, 34, 42, 23, 40, 56, 31, 8};
		//Comparable[] arr = {"d", "a", "h", "f", "g", "e", "s", "d"};
		//int[][] arr = {{73,35,1}, {5,89,30}, {69,23,100}};
		//System.out.println("The index is: " + test.sortAndSearch(arr, "s"));
		//System.out.println("The median is: " + test.sortAndSearch(arr, 25));
		//System.out.println("And the sorted array is: \n" + Arrays.toString(arr));
		//System.out.println("And the sorted array is: \n" + Arrays.deepToString(arr));
	}

	
	public Color getColor() {
		return Color.green;
	}

	public String getSpriteName() {
		return BLANKA;
	}

	public double sortAndGetMedian(int[] random) {
		int m = 0;
		int n = random.length;
			// One by one move boundary of unsorted subarray
	        for (int i = 0; i < n-1; i++)
	        {
	        	// Find the minimum element in unsorted array
	        	int min = i;
	        	for (int j = i+1; j < n; j++)
	                if (random[j] < random[min])
	                    min = j;
	 
	            // Swap the found minimum element with the first element
	            int temp = random[min];
	            random[min] = random[i];
	            random[i] = temp;
	        }
	        
	        if (random.length % 2 == 1) {
	        	m = random[(random.length)/2];
	        } else if (random.length % 2 == 0){
	        	m = (random[random.length/2] + random[(random.length/2)-1])/2;
	        }
	      
	        return m;
	}

	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {
		int n = strings.length;
		String temp = "";
		int found = 0;
		
        for (int i = 0; i < n-1; i++)
        {
        	// a negative value if this String is first, a positive value if other is first and 0 if the two objects are equal value
        	int min = i;
        	for (int j = i+1; j < n; j++)
                if (strings[j].compareTo(strings[min]) < 0) {
                    min = j;
                } 
 
            temp = strings[min];
            strings[min] = strings[i];
            strings[i] = temp;
        }
        for (int a = 0; a < strings.length; a++) {
        	if (strings[a] == toFind) {
        		found = a;
        	}
        }
		return found;
	}
	
	public double mostlySortAndGetMedian(int[] mostlySorted) {
		int temp;
		int m = 0;
        for (int i = 1; i < mostlySorted.length; i++) {
            for(int j = i ; j > 0 ; j--){
                if(mostlySorted[j] < mostlySorted[j-1]){
                    temp = mostlySorted[j];
                    mostlySorted[j] = mostlySorted[j-1];
                    mostlySorted[j-1] = temp;
                }
            }
        }
        
        if (mostlySorted.length % 2 == 1) {
        	m = mostlySorted[(mostlySorted.length)/2];
        } else if (mostlySorted.length % 2 == 0){
        	m = (mostlySorted[mostlySorted.length/2] + mostlySorted[(mostlySorted.length/2)-1])/2;
        }
      
        return m;
	}

	@Override
	public double sortMultiDim(int[][] grid) {
		int sum = 0;
		double m = 0;
		double[] medians = new double[grid.length];
		for(int i = 0; i < grid.length; i++) {
			sortAndGetMedian(grid[i]);
			medians[i] = sortAndGetMedian(grid[i]);
		}
		for (int i = 0; i < medians.length-1; i++){
	        int min = i;
	        for (int j = i+1; j < medians.length-1; j++)
	        	if (medians[j] < medians[min])
	        		min = j;
	            double temp = medians[min];
	            medians[min] = medians[i];
	            medians[i] = temp;
	        }
	        
	        if (medians.length % 2 == 1) {
	        	m = medians[(medians.length)/2];
	        } else if (medians.length % 2 == 0){
	        	m = (medians[medians.length/2] + medians[(medians.length/2)-1])/2;
	        }
	      
		return m;
	}

	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		//sort comparable and return index of comparable.
		int n = arr.length;
		int found = 0;
			// One by one move boundary of unsorted subarray
	        for (int i = 0; i < n-1; i++)
	        {
	        	// Find the minimum element in unsorted array
	        	int min = i;
	        	for (int j = i+1; j < n; j++)
	                if (arr[j].compareTo(arr[min]) < 0)
	                    min = j;
	 
	            // Swap the found minimum element with the first element
	            Comparable temp = arr[min];
	            arr[min] = arr[i];
	            arr[i] = temp;
	        }
	        for (int a = 0; a < arr.length; a++) {
	        	if (arr[a] == toFind) {
	        		found = a;
	        	}else {
	        		found = -1;
	        	}
	        }
			return found;
	}

	
}
