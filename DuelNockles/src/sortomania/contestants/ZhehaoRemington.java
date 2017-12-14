package sortomania.contestants;

import java.awt.Color;

import sortomania.Contestant;

public class ZhehaoRemington extends Contestant {
	
	
	private int[] numbers;
	private int[] test;
	private int number;

	public static void main(String[] args) {
		int[] test=new int[100];
		ZhehaoRemington testing= new ZhehaoRemington();
		populate(test);
		System.out.println("median: "+ testing.mostlySortAndGetMedian(test)+" average time is "+getTime(test));
	for(int i=0;i<test.length;i++) {
			System.out.println(test[i]);
		}
		//getTime(test);
	//	printArray(test);
		
	}
	private static long getTime(int[] arr) {
		long num=0;
		//for(int i=0;i<200;i++) {
			ZhehaoRemington useless=new ZhehaoRemington();
			long startTime= System.nanoTime();
			double med=useless.mostlySortAndGetMedian(populate(arr));
			long endTime=System.nanoTime();
			long duration=(endTime-startTime)/100000;
			num+=duration;
		//}
		return (long) num;//Math.sqrt(num/201);
		
		 
		
	}
	private static int[] populate(int[] arr) {
		for(int i=0;i<arr.length;i++) {
			arr[i]= (int)(Math.random()*99);
		}
		return arr;
	}
	

	@Override
	public Color getColor() {
		// TODO Auto-generated method stub
		return new Color(69,69,69);
	}

	@Override
	public String getSpriteName() {
		// TODO Auto-generated method stub
		return DEE_JAY;
	}
	public void sort(int[] values) {
        // check for empty or null array
        if (values ==null || values.length==0){
            return;
        }
        this.numbers = values;
        number = values.length;
        quicksort(0, number - 1);
    }

    private void quicksort(int low, int high) {
        int i = low, j = high;
        // Get the pivot element from the middle of the list
        int pivot = numbers[low + (high-low)/2];

        // Divide into two lists
        while (i <= j) {
            // If the current value from the left list is smaller than the pivot
            // element then get the next element from the left list
            while (numbers[i] < pivot) {
                i++;
            }
            // If the current value from the right list is larger than the pivot
            // element then get the next element from the right list
            while (numbers[j] > pivot) {
                j--;
            }

            // If we have found a value in the left list which is larger than
            // the pivot element and if we have found a value in the right list
            // which is smaller than the pivot element then we exchange the
            // values.
            // As we are done we can increase i and j
            if (i <= j) {
                exchange(i, j);
                i++;
                j--;
            }
        }
        // Recursion
        if (low < j)
            quicksort(low, j);
        if (i < high)
            quicksort(i, high);
    }
    public int[] insertionSort(int[] arr) {
    	int temp;
        for (int i = 1; i < arr.length; i++) {
            for(int j = i ; j > 0 ; j--){
                if(arr[j] < arr[j-1]){
                    temp = arr[j];
                    arr[j] = arr[j-1];
                    arr[j-1] = temp;
                }
            }
        }
        return arr;
    }

    private void exchange(int i, int j) {
        int temp = numbers[i];
        numbers[i] = numbers[j];
        numbers[j] = temp;
    }
    
    public static void printArray(int arr[])
    {
        int n = arr.length;
        for (int i=0; i<n; ++i)
            System.out.print(arr[i]+" ");
        System.out.println();
    }

	public double sortAndGetMedian(int[] random) {
		sort(random);
		if(random.length%2==0) {
			return (double) (random[random.length/2]+ random[random.length/2-1])/2;
		}else {
			return (double) random[random.length/2];
		}
		
	
		
	}
	public void mergeSort(String[] arr) 
	{
        if (arr.length >= 2)
        {
            String[] left = new String[arr.length / 2];
            String[] right = new String[arr.length-arr.length / 2];

            for (int i = 0; i < left.length; i++)
            {
                left[i] = arr[i];
            }
            for (int i = 0; i < right.length; i++)
            {
                right[i] = arr[i + arr.length / 2];
            }
            mergeSort(left);
            mergeSort(right);

            merge(arr, left, right);
        }
    }
	public static void merge(String[] arr, String[] left, String[] right) {
        int a = 0;
        int b = 0;
        for (int i = 0; i < arr.length; i++) {
            if (b >= right.length || (a < left.length && left[a].compareToIgnoreCase(right[b]) < 0)) {
                arr[i] = left[a];
                a++;
            } else {
                arr[i] = right[b];
                b++;
            }
        }
    }

	@Override
	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {
		mergeSort(strings);
		
		for (int i = 0; i < strings.length; i++)
		{
			if (strings[i].equals(toFind))
			{
				return i;
			}
		}
		
		return -1;
	}

	@Override
	public double mostlySortAndGetMedian(int[] mostlySorted) {
		insertionSort(mostlySorted);
		if(mostlySorted.length%2==0) {
			return (double) (mostlySorted[mostlySorted.length/2]+ mostlySorted[mostlySorted.length/2-1])/2;
		}else {
			return (double) mostlySorted[mostlySorted.length/2];
		}
	}

	@Override
	public double sortMultiDim(int[][] grid) {
		int medofmed=0;
		for(int i=0;i<grid[0].length;i++) {
			medofmed+=sortAndGetMedian(grid[i]);
		}
		return medofmed/grid[0].length;
	}

	@Override
	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		insertionSort(arr);
		for (int i = 0; i < arr.length; i++)
		{
			if (arr[i].compareTo(toFind)==0)
			{
				return i;
			}
		}
		return -1;
	}
	
	public static void insertionSort(Comparable[] arr)
	{
		int index = 0;
		for (int i = 0; i < arr.length - 1; i++)
		{
			if (arr[i].compareTo(arr[i + 1]) > 0)
			{
				index = i + 1;
				for (int j = i; j > -1; j--)
				{
					if (arr[index].compareTo(arr[j])<0)
					{
						swap(arr, index, j);
						index = j;
					}
				}
			}
		}
	}
	private static void swap(Comparable[] arr, int i, int j) {
		Comparable temp = arr[i];
		arr[i] = arr[j];
		arr[j] = temp;
	}
	private void swapNumbers(int i, int j, Comparable[] array) {
			Comparable temp;
	        temp =  array[i];
	        array[i] = array[j];
	        array[j] = temp;
	}

}

