package sortomania.contestants;

import java.awt.Color;
import java.util.Arrays;

import sortomania.Contestant;

public class AbidAbedContestant extends Contestant
{
	@Override
	public Color getColor() 
	{
		return new Color(240,60,120);
	}

	@Override
	public String getSpriteName() 
	{
		return CAMMY;
	}
	
	@Override
	public double sortAndGetMedian(int[] random) 
	{
		quickSort(random,0,random.length-1);
		return median(random);
		
	}
	@Override
	public int sortAndSearch(Comparable[] arr, Comparable toFind) 
	{
		quickSort(arr,0,arr.length-1);
		for(int i = 0; i < arr.length;i++)
		{
			if(arr[i].equals(toFind))
			{
				return i;
			}
		}
		return -1;
		
	} 
	@Override
	public int sortAndGetResultingIndexOf(String[] strings, String toFind) 
	{
		quickSort(strings,0,strings.length-1);
		for(int i = 0; i < strings.length;i++)
		{
			if(strings[i].equals(toFind))
			{
				return i;
			}
		}
		return -1;
	}

	@Override
	public double mostlySortAndGetMedian(int[] mostlySorted) 
	{
		insertionSort(mostlySorted);
		return median(mostlySorted);
	}

	@Override
	public double sortMultiDim(int[][] grid) 
	{
		double medians[] = new double[grid.length];
		for(int i=0;i < grid.length;i++) 
		{
			medians[i] = sortAndGetMedian(grid[i]);
		}
		return sortAndGetMedian(medians);
	}

	private double sortAndGetMedian(double[] random) 
	{
		quickSort(random,0,random.length-1);
		return median(random);
	}
	private double median(double[] random) 
	 {
    	if(random.length % 2 == 1)
		{
			return random[(random.length-1)/2];
		}
		else if(random.length % 2 == 0)
		{
			int x = (random.length)/2;
			return (double)(random[x] + random[x-1])/2;
		}
		else
		{
			return -1;
		}
    	
    }
	private void quickSort(double[] arr, int low, int high) 
	{
			
			if (low >= high)
				return;
	 
			int middle = low + (high - low) / 2;
			int pivot = (int) arr[middle];
	 
			int i = low;
			int j = high;
			while (i <= j) 
			{
				while (arr[i] < pivot) 
				{
					i++;
				}
	 
				while (arr[j] > pivot)
				{
					j--;
				}
	 
				if (i <= j) 
				{
					int temp = (int) arr[i];
					arr[i] = arr[j];
					arr[j] = temp;
					i++;
					j--;
				}
			}
	 
			if (low < j)
				quickSort(arr, low, j);
	 
			if (high > i)
				quickSort(arr, i, high);
			
		
	}
	public void insertionSort(int array[]) 
	{
        int n = array.length;
        for (int j = 1; j < n; j++) 
        {
            int key = array[j];
            int i = j-1;
            while ( (i > -1) && ( array [i] > key ) ) 
            {
                array [i+1] = array [i];
                i--;
            }
            array[i+1] = key;
            
        }
    }
	public void quickSort(int[] arr, int low, int high) 
	{
		
		if (low >= high)
			return;
 
		int middle = low + (high - low) / 2;
		int pivot = arr[middle];
 
		int i = low;
		int j = high;
		while (i <= j) 
		{
			while (arr[i] < pivot) 
			{
				i++;
			}
 
			while (arr[j] > pivot)
			{
				j--;
			}
 
			if (i <= j) 
			{
				int temp = arr[i];
				arr[i] = arr[j];
				arr[j] = temp;
				i++;
				j--;
			}
		}
 
		if (low < j)
			quickSort(arr, low, j);
 
		if (high > i)
			quickSort(arr, i, high);
	}	
    
    private double median(int[] random)
    {
    	if(random.length % 2 == 1)
		{
			return random[(random.length-1)/2];
		}
		else if(random.length % 2 == 0)
		{
			int x = (random.length)/2;
			return (double)(random[x] + random[x-1])/2;
		}
		else
		{
			return -1;
		}
    	
    }
    public void quickSort(String[] a, int start, int end)
    {
            int i = start;
            int j = end;

           
            if (j - i >= 1)
            {
              
                String pivot = a[i];
              
                while (j > i)
                {
                    while (a[i].compareTo(pivot) <= 0 && i < end && j > i){
                        i++;
                    }
          
                    while (a[j].compareTo(pivot) >= 0 && j > start && j >= i){
                        j--;
                    }
                  
                    if (j > i)
                        swap(a, i, j);
                }
                swap(a, start, j);
                quickSort(a, start, j - 1);
                quickSort(a, j + 1, end);
            }
        }
   private void swap(String[] a, int i, int j)
   {
        String temp = a[i];
        a[i] = a[j];
        a[j] = temp;
   }


   public void quickSort(Comparable[] a, int lo, int hi) 
   {
       if(lo >= hi) return;
       int pi = partition(a, lo, hi);
       quickSort(a, lo, pi-1);
       quickSort(a, pi+1, hi);
   }

   private int partition(Comparable[] a, int lo, int hi) 
   {
       int i = lo + 1;
       int j = hi;

       while(i <= j) {
           if(a[i].compareTo(a[lo]) <= 0) { 
               i++; 
           }
           else if(a[j].compareTo(a[lo]) > 0) { 
               j--;
           }
           else if(j < i) {
               break;
           }
           else
               swap(a, i, j);
       }
       swap(a, lo, j);
       return j;
   }
   private void swap(Comparable[] a, int i, int j) 
   {
	
	   Comparable temp = a[i];
       a[i] = a[j];
       a[j] = temp;
	
   }

}
