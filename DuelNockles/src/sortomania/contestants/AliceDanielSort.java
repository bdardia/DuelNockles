package sortomania.contestants;

import java.awt.Color;
import java.util.Arrays;


import sortomania.Contestant;

public class AliceDanielSort extends Contestant{
	//private static int size_threshold = 16;

	public static void main(String args[]) {


	}


	@Override
	public Color getColor() {
		return new Color(63,0,40);
	}

	@Override
	public String getSpriteName() {
		return KEN;
	}

	@Override
	public double sortAndGetMedian(int[] random) {
		sort(random);


		if (random.length % 2 == 0) {
			return (double)((random[random.length/2] + random[random.length/2 - 1]) / 2.0);
		}
		return (double)(random[(random.length-1) / 2]);
	}

	@Override
	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {

		String[] strings2 = mergeSort(strings);
		for(int i = 0; i < strings2.length;i++)
		{
			strings[i] = strings2[i];
		}
		for(int x = 0; x < strings.length;x++)
		{
			if(toFind.equals(strings[x]))
			{
				return x;
			}
		}
		return -1;
	}

	@Override
	public double mostlySortAndGetMedian(int[] mostlySorted) {
		return sortAndGetMedian(mostlySorted);

	}

	@  Override
	public double sortMultiDim(int[][] grid) {
		int[] medianArr = new int[grid[0].length];
		for (int x = 0; x < grid.length; x++) {
			medianArr[x]= (int) sortAndGetMedian(grid[x]);
		}
		int[] medianArrPos = new int[medianArr.length];
		for (int y = 0; y < medianArrPos.length; y++) {
			medianArrPos[y] = y;
		}
		bubbleSort(medianArr, medianArrPos);

		for(int i = 0; i < medianArrPos.length; i++)
		{
			int[] temp = new int [grid[0].length];
			temp = grid[i];
			grid[i] = grid[medianArrPos[i]];
			grid[medianArrPos[i]] = temp;
		}


		for(int x = 0;x < grid[0].length; x++){
			for(int y = 0; y < grid.length/2; y++) {
				int temp = grid[y][x];
				grid[y][x] = grid[grid.length - y - 1][x];
				grid[grid.length - y - 1][x] = temp;
			}
		}
		return sortAndGetMedian(medianArr);
	}

	@Override
	public int sortAndSearch(Comparable[] arr, Comparable toFind) {

		bubbleSort(arr);
		for (int i = 0; i < arr.length; i++) {
			if (arr[i].equals(toFind)) {
				return i;
			}
		}
		return -1;

	}


	public static void sort(int[] a)
	{
		introsort_loop(a, 0, a.length, 2*floor_lg(a.length));
	}

	public static void sort(int[] a, int begin, int end)
	{
		if (begin < end)
		{
			introsort_loop(a, begin, end, 2*floor_lg(end-begin));
		}
	}

	private static void introsort_loop (int[] a, int lo, int hi, int depth_limit)
	{
		while (hi-lo > (a.length -1))
		{
			if (depth_limit == 0)
			{
				heapsort(a, lo, hi);
				return;
			}
			depth_limit=depth_limit-1;
			int p=partition(a, lo, hi, medianof3(a, lo, lo+((hi-lo)/2)+1, hi-1));
			introsort_loop(a, p, hi, depth_limit);
			hi=p;
		}
		insertionsort(a, lo, hi);
	}

	private static int partition(int[] a, int lo, int hi, int x)
	{
		int i=lo, j=hi;
		while (true)
		{
			while (a[i] < x) i++;
			j=j-1;
			while (x < a[j]) j=j-1;
			if(!(i < j))
				return i;
			exchange(a,i,j);
			i++;
		}
	}

	private static int medianof3(int[] a, int lo, int mid, int hi)
	{
		if (a[mid] < a[lo])
		{
			if (a[hi] < a[mid])
				return a[mid];
			else
			{
				if (a[hi] < a[lo])
					return a[hi];
				else
					return a[lo];
			}
		}
		else
		{
			if (a[hi] < a[mid])
			{
				if (a[hi] < a[lo])
					return a[lo];
				else
					return a[hi];
			}
			else
				return a[mid];
		}
	}

	/*
	 * Heapsort algorithm
	 */
	private static void heapsort(int[] a, int lo, int hi)
	{
		int n = hi-lo;
		for (int i=n/2; i>=1; i=i-1)
		{
			downheap(a,i,n,lo);
		}
		for (int i=n; i>1; i=i-1)
		{
			exchange(a,lo,lo+i-1);
			downheap(a,1,i-1,lo);
		}
	}

	private static void downheap(int[] a, int i, int n, int lo)
	{
		int d = a[lo+i-1];
		int child;
		while (i<=n/2)
		{
			child = 2*i;
			if (child < n && a[lo+child-1] < a[lo+child])
			{
				child++;
			}
			if (d >= a[lo+child-1]) break;
			a[lo+i-1] = a[lo+child-1];
			i = child;
		}
		a[lo+i-1] = d;
	}

	/*
	 * Insertion sort algorithm
	 */
	private static void insertionsort(int[] a, int lo, int hi)
	{
		int i,j;
		int t;
		for (i=lo; i < hi; i++)
		{
			j=i;
			t = a[i];
			while(j!=lo && t < a[j-1])
			{
				a[j] = a[j-1];
				j--;
			}
			a[j] = t;
		}
	}

	/*
	 * Common methods for all algorithms
	 */
	private static void exchange(int[] a, int i, int j)
	{
		int t=a[i];
		a[i]=a[j];
		a[j]=t;
	}

	private static int floor_lg(int a)
	{
		return (int)(Math.floor(Math.log(a)/Math.log(2)));
	}
	public String[] mergeSort(String[] list) {
		if (list.length == 1) {
			return list;
		}
		else {
			String[] newList = Arrays.copyOfRange(list,0,list.length / 2);
			String[] newList2 = Arrays.copyOfRange(list,list.length / 2,list.length);
			return merge(mergeSort(newList),mergeSort(newList2));

		}
	}
	public String[] merge(String[] list1, String[] list2) {
		String[] list3 = new String[list1.length + list2.length];
		int x = 0;
		int y = 0;
		int index = 0;
		while(x < list1.length || y < list2.length)
		{
			if(x == list1.length) //full of x
			{
				list3[index] = list2[y];
				y++;
				index++;
			}
			else if(y == list2.length) //full of y
			{
				list3[index] = list1[x];
				x++;
				index++;
			}
			else if(list1[x].compareTo(list2[y]) < 0) // if list1's item is smaller than list2's item
			{
				list3[index]= list1[x];
				x++;
				index++;
			}
			else
			{
				list3[index]= list2[y];
				y++;
				index++;
			}
		}
		return list3;
	}
	public static void bubbleSort(int[] medianArr, int[] medianArrPos) {
		int counter = medianArr.length-1;
		while (counter > 0) {
			for (int x = 0; x < medianArr.length-1; x++) {
				for (int y = x+1; y < medianArr.length; y++) {
					if (medianArr[x] > (medianArr[y])) {
						int temp = medianArr[x];
						medianArr[x] = medianArr[y];
						medianArr[y] = temp;
						int temp2 = medianArrPos[x];
						medianArrPos[x] = medianArrPos[y];
						medianArrPos[y] = temp2;
					}
					else {
						counter--;
					}
				}
			}
		}	
	}

	public static void bubbleSort(Comparable[] arr)
	{
		boolean change = false;
		while (!change) {
			int numChanges = 0;
			for (int x = 0; x < arr.length-1; x++) {
				Comparable temp;
				if (arr[x].compareTo(arr[x+1]) > 0) {
					temp = arr[x+1];
					arr[x+1] = arr[x];
					arr[x] = temp;
					numChanges++;
				}		
			}
			if (numChanges == 0) {
				change = true;
			}
		}
	} 

}