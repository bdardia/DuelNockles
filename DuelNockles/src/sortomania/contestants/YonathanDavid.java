package sortomania.contestants;

import java.awt.Color;
import java.util.Arrays;

import sortomania.Contestant;

public class YonathanDavid extends Contestant{


	public static void main(String[] args) {
		int[] a = {2, 6, 3, 6,1,8,11,5};
		int[] b = {1,3,5,7,9,11,13,15,17};
		int[] c = {2,8,5,3,0,10,13,14,21};
		int[] d= {5,4,7,0,12,15,17,11,20};
		String[] z = {"kjgae","aergre","aaf","abc","asfff","dnreiugnge","xroigner","xae","d386fggeg"};
		int[][] x = {a,b,c,d};
		YonathanDavid test = new YonathanDavid();
		
		double y = test.sortMultiDim(x);
		for(int i=0;i<x.length;i++) {
		System.out.println(Arrays.toString(x[i]));
		}
		System.out.println(y);
		

		//double h = test.mostlySortAndGetMedian(a);
		//double g = test.sortAndGetMedian(b);
		//System.out.println(h);
		//System.out.println(Arrays.toString(a));
		//System.out.println(g);
		//System.out.println(Arrays.toString(b));
		//int r = test.sortAndGetResultingIndexOf(z,"kjgae"); 
		//System.out.println(r);
		//System.out.println(Arrays.deepToString(z));
		
		
	/*	int r = test.sortAndSearch(z, "d386fggeg");  
		System.out.println(r);
		System.out.println(Arrays.deepToString(z));
		*/
		
	}

	@Override
	public Color getColor() {
		// TODO Auto-generated method stub
		return new Color(0,0,0);
	}

	@Override
	public String getSpriteName() {
		// TODO Auto-generated method stub
		return  CAMMY;
	}

	@Override
	public double sortAndGetMedian(int[] arr) {
		//Yonathan
		sort(arr,0,arr.length-1);
		if(arr.length%2==0) {
        	return ((double)(arr[arr.length/2-1]+arr[arr.length/2])/2);
        }
        return arr[(arr.length-1)/2];
	}

	@Override
	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {
		//David
		 int n = strings.length;

	        for (int i = 0; i < n-1; i++)
	            for (int j = 0; j < n-i-1; j++)
	          
	                if (strings[j].compareTo(strings[j+1])>0)
	                {
	                	
	                    // swap temp and arr[i]
	                	//strings[j].compareTo(strings[j+1])>0
	                    String temp = strings[j];
	                    strings[j] =strings[j+1];
	                    strings[j+1] = temp;	                
	                    }
		  
		for(int i = 0;i<strings.length-1;i++) {
			if(strings[i]== toFind) {
				
				return i;
			}
		}
		return -1;
	}
	
/*
	public boolean check(String s1,String s2) {
		//check if one is larger than the other
		//run along each string checking each letter until it runs out.\
		//run the for loop for the length of shorter stirng and if for loops end without returning anything run a if statement that if the shorter string is the second string then return true;
		//return true = first string is bigger than second string
		int l;// <- shortest string
		int f = s1.length();
		int s = s2.length();
		if(f>s) {
			l = s;
		}else
		{
			l =f;
		}
		for(int i =0;i<l-1;i++) {
			if(s1.substring(i,i+1).compareTo(s2.substring(i,i+1))>0) {
				return true;
			}
		}
		if(s == l) {
			return true;
		}

		return false;
	}
*/
	@Override
	public  double mostlySortAndGetMedian(int[] arr) {
		// make sure median is returned correctly
		        int n = arr.length;
		        for (int i = 0; i < n-1; i++)
		            for (int j = 0; j < n-i-1; j++)
		                if (arr[j] > arr[j+1])
		                {
		                    // swap temp and arr[i]
		                    int temp = arr[j];
		                    arr[j] =arr[j+1];
		                    arr[j+1] = temp;
		                }
		        if(arr.length%2==0) {
		        	return ((double)(arr[arr.length/2-1]+arr[arr.length/2])/2);
		        }
		        return arr[(arr.length-1)/2];
		    }
	


	@Override
	public double sortMultiDim(int[][] grid) {
		// David
		double[] midz = new double[grid.length];
		for(int row =0;row<grid.length;row++) {
			double y =sortAndGetMedian(grid[row]);
			midz[row] = y;
			//System.out.println(y);
		}
		if(midz.length%2==0) {
			//System.out.println( ((double)(midz[midz.length/2-1]+midz[midz.length/2])/2));
        	return ((double)(midz[midz.length/2-1]+midz[midz.length/2])/2);
        }
		//System.out.println(midz[(midz.length-1)/2]);
        return midz[(midz.length-1)/2];
        
	}

	@Override
	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		 int n = arr.length;

	        for (int i = 0; i < n-1; i++)
	            for (int j = 0; j < n-i-1; j++)
	          
	                if (arr[j].compareTo(arr[j+1])>0)
	                {
	                	
	                    // swap temp and arr[i]
	                	//strings[j].compareTo(strings[j+1])>0
	                    Comparable temp = arr[j];
	                    arr[j] =arr[j+1];
	                    arr[j+1] = temp;	                
	                    }
		  
		for(int i = 0;i<arr.length-1;i++) {
			if(arr[i]== toFind) {
				
				return i;
			}
		}
		return -1;
		
	}
	public void merge(int arr[], int l, int m, int r)
    {
        // Find sizes of two subarrays to be merged
        int n1 = m - l + 1;
        int n2 = r - m;
 
        /* Create temp arrays */
        int L[] = new int [n1];
        int R[] = new int [n2];
 
        /*Copy data to temp arrays*/
        for (int i=0; i<n1; ++i)
            L[i] = arr[l + i];
        for (int j=0; j<n2; ++j)
            R[j] = arr[m + 1+ j];
 
 
        /* Merge the temp arrays */
 
        // Initial indexes of first and second subarrays
        int i = 0, j = 0;
 
        // Initial index of merged subarry array
        int k = l;
        while (i < n1 && j < n2)
        {
            if (L[i] <= R[j])
            {
                arr[k] = L[i];
                i++;
            }
            else
            {
                arr[k] = R[j];
                j++;
            }
            k++;
        }
 
        /* Copy remaining elements of L[] if any */
        while (i < n1)
        {
            arr[k] = L[i];
            i++;
            k++;
        }
 
        /* Copy remaining elements of R[] if any */
        while (j < n2)
        {
            arr[k] = R[j];
            j++;
            k++;
        }
    }
 
    // Main function that sorts arr[l..r] using
    // merge()
   public void sort(int arr[], int l, int r)
    {
        if (l < r)
        {
            // Find the middle point
            int m = (l+r)/2;
 
            // Sort first and second halves
            sort(arr, l, m);
            sort(arr , m+1, r);
 
            // Merge the sorted halves
            merge(arr, l, m, r);
        }
    }


}

