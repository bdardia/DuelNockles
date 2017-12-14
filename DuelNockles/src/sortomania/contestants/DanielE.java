package sortomania.contestants;

import java.awt.Color;
import java.util.Arrays;
import sortomania.Contestant;

public class DanielE extends Contestant {

	@Override
	public Color getColor() {
		return new Color(25, 162, 43); 
	}

	@Override
	public String getSpriteName() {
		return CHUN_LI;
	}

	@Override
	public double sortAndGetMedian(int[] random) {
		quickSort(random, 0, random.length);
		return (int) median(random);
	}

	private double median(int[] random) {
		if (random.length % 2 == 0) {
			return (double)((random[random.length/2] + random[random.length/2 - 1]) / 2.0);
		}
		return (double)(random[(random.length-1) / 2]);
	}

	private void quickSort(int[] random, int i, int length) {
		if(length - i <= 1)
		{
			return;
		}
		else {
			double pivotPos = partition(random, i, length)-1;	
			quickSort(random, i, (int) pivotPos);
			quickSort(random, (int) (pivotPos+1), length); 
		}
		
	}

	private double partition(int[] random, int i, int length) {
		double pivot = random[i];
		int front2 = i;
		int pivotPos = i;
		for (int x = i+1; x < length; x++) 
		{
			if (random[x] <= pivot)
			{
				if (pivotPos == front2) 
				{
					pivotPos = x;
				}
				double temp = random[front2];
				random[front2] = random[x];
				random[x] = (int) temp;
				front2++;
			}	
		}
		double temp = random[pivotPos];
		random[pivotPos] = random[front2];
		random[front2] = (int) temp;
		return front2+1;
	}

	@Override
	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {
		String[] array = mergeSort(strings);
		for(int i = 0; i < array.length;i++){
			strings[i] = array[i];
		}
		for(int x = 0; x < strings.length;x++){
			if(toFind.equals(strings[x])){
				return x;
			}
		}
		return -1;	
	}

	private String[] mergeSort(String[] strings) {
		if (strings.length == 1) {
			return strings;
		}
		else {
			String[] newList = Arrays.copyOfRange(strings,0,strings.length / 2);
			String[] newList2 = Arrays.copyOfRange(strings,strings.length / 2,strings.length);
			return merge(mergeSort(newList),mergeSort(newList2));
			
		}
	}

	private String[] merge(String[] mergeSort, String[] mergeSort2) {
		String[] merged = new String[mergeSort.length + mergeSort2.length];
		int x = 0;
		int y = 0;
		int index = 0;
		while(x < mergeSort.length || y < mergeSort2.length)
		{
			if(x == mergeSort.length) {
				merged[index] = mergeSort2[y];
				y++;
				index++;
			}
			else if(y == mergeSort2.length)	{
				merged[index] = mergeSort[x];
				x++;
				index++;
			}
			else if(mergeSort[x].compareTo(mergeSort2[y]) < 0) 	{
				merged[index]= mergeSort[x];
				x++;
				index++;
			}
			else{
				merged[index]= mergeSort2[y];
				y++;
				index++;
			}
		}
		return merged;
	}

	@Override
	public double mostlySortAndGetMedian(int[] mostlySorted) {
		return sortAndGetMedian(mostlySorted);
	}

	@Override
	public double sortMultiDim(int[][] grid) {
		int[] newArray = new int[grid[0].length];
		for (int x = 0; x < grid.length; x++) {
			newArray[x]= (int) sortAndGetMedian(grid[x]);
		}
		int[] x = new int[newArray.length];
		for (int y = 0; y < x.length; y++) {
			x[y] = y;
		}
		bubbleSort(newArray, x);
		for(int i = 0; i < x.length; i++)
		{
			int[] temp = new int [grid[0].length];
			temp = grid[i];
			grid[i] = grid[x[i]];
			grid[x[i]] = temp;
		}
		reverseC(grid);
		return sortAndGetMedian(newArray);
	}

	private void reverseC(int[][] grid) {
		for(int x = 0;x < grid[0].length; x++){
            for(int y = 0; y < grid.length/2; y++) {
                int temp = grid[y][x];
                grid[y][x] = grid[grid.length - y - 1][x];
                grid[grid.length - y - 1][x] = temp;
            }
		}
	}

	public void bubbleSort(int[] newArray, int[] position) {
			int counter = newArray.length-1;
			while (counter > 0) {
				for (int x = 0; x < newArray.length-1; x++) {
					for (int y = x+1; y < newArray.length; y++) {
						if (newArray[x] > (newArray[y])) {
							int temp = newArray[x];
							newArray[x] = newArray[y];
							newArray[y] = temp;
							int temp2 = position[x];
							position[x] = position[y];
							position[y] = temp2;
						}
						else {
							counter--;
						}
					}
				}
			}	
		}
	@Override
	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		Comparable[] arr2 = mergeSortTask5(arr);
		for(int i = 0; i < arr2.length;i++)
		{
			arr[i] = arr2[i];
		}
		for(int x = 0;x < arr.length;x++)
		{
			if(arr2[x].equals(toFind))
			{
				return x;
			}
		}
		return -1;
	}

	public Comparable[] mergeSortTask5(Comparable[] list) {
		if (list.length == 1) {
			return list;
		}
		else {
			Comparable[] newList = Arrays.copyOfRange(list,0,list.length / 2);
			Comparable[] newList2 = Arrays.copyOfRange(list,list.length / 2,list.length);
			return merge2(mergeSortTask5(newList),mergeSortTask5(newList2));
			
		}
	}

	public Comparable[] merge2(Comparable[] list1, Comparable[] list2) {
		Comparable[] merged = new Comparable[list1.length + list2.length];
		int x = 0;
		int y = 0;
		int index = 0;
		while(x < list1.length || y < list2.length)
		{
			if(x == list1.length) //full of x
			{
				merged[index] = list2[y];
				y++;
				index++;
			}
			else if(y == list2.length) //full of y
			{
				merged[index] = list1[x];
				x++;
				index++;
			}
			else if(list1[x].compareTo(list2[y]) < 0) // if list1's item is smaller than list2's item
			{
				merged[index]= list1[x];
				x++;
				index++;
			}
			else
			{
				merged[index]= list2[y];
				y++;
				index++;
			}
		}
		return merged;
	}

}
