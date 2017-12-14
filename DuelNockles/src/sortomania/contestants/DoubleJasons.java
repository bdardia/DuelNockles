package sortomania.contestants;

import java.awt.Color;

import sortomania.Contestant;

public class DoubleJasons extends Contestant implements Comparable{
	
	private int[] ISSeq = {1,3,7,21,48,112,336,861,1968,4592};
	private static String[] names = {"Tashia","Fidela","Carley","starla","Maisie","Elijah","pearl","Jacquie","Zelma","Tama","Hannelore","Shaniqua","Isa","Emily","Desiree","Garnet","Lauri","Erna","Denese","Renato","Britney","Numbers","Randa","Jewel","Vincenzo","Arianna","Johnathon","Charlette","Rae","Jerald"};
	private static int[][] nums = {{1,2312,12312,35435,345357},{7767,657,6765,76,576},{5765765,765,76,57,657,65,765,765},{765,7,657,657569},{789,79,907,78,9,879,879,78}};
	public DoubleJasons() {															  //57,65,76,657,765,765,765,5765765	
	}
	
	public static final void main(String[] args) {
		DoubleJasons test = new DoubleJasons();
		test.shellSort(names);
		for(String name : names) {
			System.out.println(name);
		}
	}
	
	private void shellSort(String[] arr) {
		int index = ISSeq.length-1;
		int gap = ISSeq[index];
		String holder;
		String char1;
		String char2;
		for(int i = 0; i< arr.length; i++) //iterate through arr
		{
			if(i+ gap >= arr.length)
			{
				i = -1;
				index --;
				if(index ==-1) break;
				gap = ISSeq[index];
			}
			else 
			{
				for(int idx = 0; idx< arr[i].length()-1; idx++)
				{
					char1 = arr[i].substring(idx, idx+1).toLowerCase(); 
					char2 = arr[i+gap].substring(idx, idx+1).toLowerCase();
					if( char1.compareTo(char2) > 0)
					{
						holder = arr[i];
						arr[i] = arr[i+gap];
						arr[i+gap] = holder;
						checkPreviousIndices(i,gap,arr);
						break;
					}
					if(char1.compareTo(char2) < 0)break;
				}
			}	
		}
	}
	
	private void shellSort(int[] arr) {
		int index = ISSeq.length-1;
		int gap = ISSeq[index];
		int holder;
		for(int i = 0; i< arr.length; i++) //iterate through arr
		{
			if(i+gap >= arr.length)
			{
				i = -1;
				index --;
				if(index ==-1) break;
				gap = ISSeq[index];
			}
			else 
			{
				if(arr[i] > arr[i+gap]) {
					holder = arr[i];
					arr[i] = arr[i+gap];
					arr[i+gap] = holder;
					checkPreviousIndices(i,gap,arr);
				}
			}
		}
	}

	private void shellSort(double[] arr) {
		int index = ISSeq.length-1;
		int gap = ISSeq[index];
		double holder;
		while( index > 0) {
			for(int i = 0; i< arr.length; i++) //iterate through arr
			{
				if(i+ gap >= arr.length)
				{
					i = -1;
					index --;
					if(index ==-1) break;
					gap = ISSeq[index];
				}
				else 
				{
					if(arr[i] > arr[i+gap]) {
						holder = arr[i];
						arr[i] = arr[i+gap];
						arr[i+gap] = holder;
						checkPreviousIndices(i,gap,arr);
					}
				}
			}
		}
	}
	
	private void checkPreviousIndices(int index, int gap, String[] arr) {
		while(index - gap >= 0) {
			for(int secondCheck = 0; secondCheck< arr[index].length()-1; secondCheck++)
			{
				String char1 = arr[index-gap].substring(secondCheck, secondCheck+1).toLowerCase();
				String char2 = arr[index].substring(secondCheck, secondCheck+1).toLowerCase();
				if(char1.compareTo(char2) > 0)
				{
					String holder = arr[index-gap];
					arr[index-gap] = arr[index];
					arr[index] = holder;
					index -= gap;
					break;
				}
				if(char1.compareTo(char2) <= 0) return;
			}
		}
	}
	
	private void checkPreviousIndices(int index, int gap, int[] arr) {
		while(index - gap >= 0) {
			if(arr[index] < arr[index - gap])
			{
				int holder = arr[index-gap];
				arr[index-gap] = arr[index];
				arr[index] = holder;
				index -= gap;
			}
			else return;
		}
	}
	
	private void checkPreviousIndices(int index, int gap, double[] arr) {
		while(index - gap >= 0) {
			if(arr[index] < arr[index - gap])
			{
				double holder = arr[index-gap];
				arr[index-gap] = arr[index];
				arr[index] = holder;
				index -= gap;
			}
			else return;
		}
	}
	
	@Override
	public Color getColor() {
		return new Color(255,0,255);
	}

	@Override
	public String getSpriteName() {
		return CHUN_LI;
	}

	@Override
	public double sortAndGetMedian(int[] random) {	
		shellSort(random);
		double half = ((double)random.length-1)/2;
		return (random[(int) Math.floor(half)] + random[(int) Math.ceil(half)])/2;
	}

	public double sortAndGetMedian(double[] medians) {
		shellSort(medians);
		double half = ((double)medians.length-1)/2;
		return (medians[(int) Math.floor(half)] + medians[(int) Math.ceil(half)])/2;
	}
	
	@Override
	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {
		shellSort(strings);
		for(int i = 0; i < strings.length; i++) {
			if(strings[i].equals(toFind)) return i;
		}
		return -1;
	}

	@Override
	public double mostlySortAndGetMedian(int[] mostlySorted) {
		shellSort(mostlySorted);
		double half = (double)((mostlySorted.length-1)/2);
		return (mostlySorted[(int) Math.floor(half)] + mostlySorted[(int) Math.ceil(half)])/2;
	}

	@Override
	public double sortMultiDim(int[][] grid) {
		double[] medians = new double[grid.length];
		for (int i = 0; i < grid.length; i++) {
			medians[i] = sortAndGetMedian(grid[i]);
		}
		return sortAndGetMedian(medians);
	}

	@Override
	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		shellSort(arr);
		for( int i =0; i< arr.length; i++)
		{
			if( arr[i].compareTo(toFind) == 0)
			{
				return i;
			}
		}
		return -1;
	}

	private void shellSort(Comparable[] arr) {
		int index = ISSeq.length-1;
		int gap = ISSeq[index];
		Comparable holder;
		while( index > 0) {
			for(int i = 0; i< arr.length; i++) //iterate through arr
			{
				if(i+ gap >= arr.length)
				{
					i = -1;
					index --;
					if(index ==-1) break;
					gap = ISSeq[index];
				}
				else 
				{
					if(arr[i].compareTo(arr[i+gap]) > 0) {
						holder = arr[i];
						arr[i] = arr[i+gap];
						arr[i+gap] = holder;
						checkPreviousIndices(i,gap,arr);
					}
				}
			}
		}
	}

	private void checkPreviousIndices(int i, int gap, Comparable[] arr) {
		while(i - gap >= 0) {
			if(arr[i].compareTo(arr[i - gap]) > 0)
			{
				Comparable holder = arr[i-gap];
				arr[i-gap] = arr[i];
				arr[i] = holder;
				i -= gap;
			}
			else return;
		}
	}

	@Override
	public int compareTo(Object arg) {
		return (this.toString().compareTo(arg.toString()))*(-1);
	}


}
