package sortomania.contestants;

import java.awt.Color;
import java.util.Arrays;

import sortomania.Contestant;

public class VickieAreej extends Contestant{

	private double median;
	
	public VickieAreej() {
	}

	public static void main(String[] args) {
		VickieAreej run = new VickieAreej();
		
		/*int[] arr = {4, 7, 35,234,68,8765,453,345345,234,756,242,546,2,6,7,34,42,5,345,7,4};
		int[][] arr2 = {{2,1,3,98,9,45,345,2,672,23},{23,54,23,8,3,234,3,6,23,67}};

		//System.out.print(Arrays.toString(run.multiArraySort2(arr2)));//1d arr
		//System.out.print(Arrays.deepToString(arr2)); //2d arr
		
		System.out.print("TEST 1:\n");
		System.out.println("	The median is: " + run.sortAndGetMedian(arr));
		System.out.print("	"+Arrays.toString(arr));//1d arr
		System.out.println("\n----------------------------------------------------------------------------------------");
		
		
		System.out.print("TEST 2:\n");
		String[ ] names = {"joe", "SLIM", "ed", "george", "Vivkie", "vickie", "slim", "dfgdf", "erter", "erterter"};
		run.sortAndGetResultingIndexOf(names, "vickie");
		System.out.print(run.sortAndGetResultingIndexOf(names, "vickie"));//1d arr
		System.out.print("	"+Arrays.toString(names));//1d arr
		System.out.println("\n----------------------------------------------------------------------------------------");
		
		
		System.out.print("TEST 3:\n");
		int[] ar = {4, 7,10, 2,18,12, 34, 42, 23, 40, 56, 31, 8, 56,234,56,75,25,7,5,456,25,35,234,67,5,23,4,345,4,5,24,23,4,5,76,23,1,4,654,234,54};
		run.mostlySortAndGetMedian(ar);
		System.out.println("	The median is: " + run.median);
		System.out.print("	"+Arrays.toString(ar));//1d arr
		System.out.println("\n----------------------------------------------------------------------------------------");
		
		System.out.print("TEST 4:\n");
		System.out.println("	The median is: " + run.sortMultiDim(arr2));
		System.out.print("	"+Arrays.toString(run.multiArraySort2(arr2)));//1d arr
		System.out.println("\n----------------------------------------------------------------------------------------");
		
		System.out.print("TEST 5:\n");
		String[ ] names2 = {"joe", "SLIM", "ed", "george", "Vivkie", "vickie", "slim","erterwt", "qwer", "wertwerb"};
		System.out.println("The sorted string array is: \n" + run.sortAndSearch(names2, "vickie"));
		System.out.print("	"+Arrays.toString(names2));
		System.out.println("\n----------------------------------------------------------------------------------------");
		*/
		
    }  

	@Override
	public Color getColor() {
		 return new Color(218,112,214);
	}

	@Override
	public String getSpriteName() {
		return FEI_LONG;
	}
    
    //TASK1
	public double sortAndGetMedian(int[] random) {
		heapSort(random);

		if(random.length%2 ==0) {
			median = ((double)random[random.length/2]+(double)random[(random.length/2)-1])/2;
		}else {
			median = (double)random[(int) ((random.length/2)+.5)];
		}
		return median;
	}
	
	//sorting algorithm 
	public void heapSort(int arr[]){
        int n = arr.length;
        for (int i = n / 2 - 1; i >= 0; i--)
            heapify(arr, n, i);
        for (int i=n-1; i>=0; i--){
            int temp = arr[0];
            arr[0] = arr[i];
            arr[i] = temp;
 
            heapify(arr, i, 0);
        }
    }
 
    public void heapify(int arr[], int n, int i){
        int largest = i; 
        int l = 2*i + 1;  
        int r = 2*i + 2; 

        if (l < n && arr[l] > arr[largest])
            largest = l;

        if (r < n && arr[r] > arr[largest])
            largest = r;

        if (largest != i){
            int swap = arr[i];
            arr[i] = arr[largest];
            arr[largest] = swap;
            heapify(arr, n, largest);
        }
    }
	 
	//TASK2
	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {
		sortStringExchange(strings);
	
		return binarySearch(strings, toFind);
	}
	
		//sort string array
	public void sortStringExchange( String x [] ){
       int i, j;
       String temp;

       for(i = 0; i<x.length-1; i++){
    	   for (j = i+1;j < x.length; j++){  
    		   if ( x[i].compareToIgnoreCase(x[j])>0 ) {                                            
    			   temp = x [ i ];
                   x [ i ] = x [ j ];  
                   x [ j ] = temp;     
               } 
           } 
        } 
     }
		 
	 public int binarySearch(String[] a, String x) {
        int low = 0;
        int high = a.length - 1;
        int mid;

        while (low <= high) {
            mid = (low + high) / 2;

            if (a[mid].compareTo(x) < 0) {
                low = mid + 1;
            } else if (a[mid].compareTo(x) > 0) {
                high = mid - 1;
            } else {
                return mid;
            }
        }

        return -1;
    }
	
	//TASK3
	public double mostlySortAndGetMedian(int[] mostlySorted) {
		bubbleSort(mostlySorted);
		
		//get median
		if(mostlySorted.length%2 ==0) {
			median = ((double)mostlySorted[mostlySorted.length/2]+(double)mostlySorted[(mostlySorted.length/2)-1])/2;
		}else {
			median = (double)mostlySorted[(int) ((mostlySorted.length/2)+.5)];
		}
		return median;
	}
	
	public int[] bubbleSort(int[] mostlySorted) {
		int n = mostlySorted.length;  
        int temp = 0;  
         for(int i=0; i < n; i++){  
        	 for(int j=1; j < (n-i); j++){  
        		 if(mostlySorted[j-1] > mostlySorted[j]){  
        			 temp = mostlySorted[j-1];  
                     mostlySorted[j-1] = mostlySorted[j];  
                     mostlySorted[j] = temp;  
                 }  
             }  
         }
         return mostlySorted;
	}

	//TASK4
	public double sortMultiDim(int[][] grid) {
		int[] list = multiArraySort2(grid);
		
		//getMedian
		if(list.length%2 ==0) {
			median = ((double)list[list.length/2]+(double)list[(list.length/2)-1])/2;
		}else {
			median = (double)list[(int) ((list.length/2)+.5)];
		}
		return median;
	}

	public int[] multiArraySort2(int[][] grid) {
	    int[] list = new int[grid.length*grid[0].length];
	    int listPos = 0;
	    for(int i = 0 ; i < grid.length; i++) {
	        for(int j = 0; j < grid[i].length; j++) {
	            list[listPos++] = grid[i][j];
	        }
	    }
	    heapSort(list);
	    return list;
	}

	//TASK5
	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		 sortComArrExchange(arr);
		
		return binarySearchComparable(arr,toFind);
	}
	
	public void sortComArrExchange( Comparable x [] ){
       int i, j;
       Comparable temp;

       for(i = 0; i<x.length-1; i++){
    	   for (j = i+1;j < x.length; j++){  
    		   if ( ((String) x[i]).compareToIgnoreCase((String) x[j])>0 ) {                                            
    			   temp = x [ i ];
                   x [ i ] = x [ j ];  
                   x [ j ] = temp;     
               } 
           } 
        } 
     }
	
	 public static int binarySearchComparable(Comparable[] a, Comparable x) {
        int low = 0;
        int high = a.length - 1;
        int mid;

        while (low <= high) {
            mid = (low + high) / 2;

            if (a[mid].compareTo(x) < 0) {
                low = mid + 1;
            } else if (a[mid].compareTo(x) > 0) {
                high = mid - 1;
            } else {
                return mid;
            }
        }
        return -1;
    }
	 
	public String toString() {
		return "A & V";
		
	}

}
