package sortomania;

import java.util.Arrays;

import javax.swing.plaf.SliderUI;

public class CompetitionRunner implements Runnable {

	Contestant c;
	DataSets ds;
	Pair pair;

	public CompetitionRunner(Contestant c, DataSets ds, Pair pair) {
		this.c= c;
		this.ds = ds;
		this.pair = pair;

	}

	@Override
	public void run() {
		int trials = ds.getTotalTrials();
		
		test(new Task() {

			@Override
			public void task(int i) {
				test1(i);
			}

			@Override
			public int getNumber() {
				// TODO Auto-generated method stub
				return 1;
			}
		}, trials);
		test(new Task() {

			@Override
			public void task(int i) {
				test2(i);
			}

			@Override
			public int getNumber() {
				// TODO Auto-generated method stub
				return 2;
			}
		}, trials);
		test(new Task() {

			@Override
			public void task(int i) {
				test3(i);
			}

			@Override
			public int getNumber() {
				// TODO Auto-generated method stub
				return 3;
			}
		}, trials);
		test(new Task() {

			@Override
			public void task(int i) {
				test4(i);
			}

			@Override
			public int getNumber() {
				// TODO Auto-generated method stub
				return 4;
			}
		}, trials);
		test(new Task() {

			@Override
			public void task(int i) {
				test5(i);
			}

			@Override
			public int getNumber() {
				// TODO Auto-generated method stub
				return 5;
			}
		}, trials);


		//pair will only allow winner to be set if it has not been set already.
		pair.setWinner(c);


	}

	public void test(Task task, int trials){
		for(int i = 0; i < trials && Runner.screen.isKeepRunning(); i ++){
			c.beginTask(task.getNumber());
			long time = System.nanoTime();
			long errorPenalty = 0;
			String errorName = "";
			int penalty = pair.getWinner() == null ? 1000 : 0;
			try{
				task.task(i);
			}catch(ArrayIndexOutOfBoundsException e){
				errorPenalty = 100000000;
				errorName = "ArrayIndexOutOfBoundsException!";
//				c.penalize("ArrayIndexOutOfBoundsException!",penalty);
//				c.successfulSort(false, i);
//				c.successfulFind(false, i);
			}catch(NullPointerException npe){
				errorPenalty = 100000000;
				errorName = "NullPointerException!";
//				c.penalize("NullPointerException!",penalty);
//				c.successfulSort(false, i);
//				c.successfulFind(false, i);
			}catch(StackOverflowError soe){
				errorPenalty = 100000000;
				errorName = "Stack Overflow Error!";
//				c.penalize("Stack Overflow Error!",penalty);
//				c.successfulSort(false, i);
//				c.successfulFind(false, i);
			}catch(Exception e){
				errorPenalty = 100000000;
				errorName = e.toString();
//				c.penalize("Unknown Error!",penalty);
//				c.successfulSort(false, i);
//				c.successfulFind(false, i);
			}finally{
				if(errorPenalty > 0){
					System.out.println("Error: "+c+":"+errorName);
					c.penalize(errorName,penalty);
					c.successfulSort(false, i);
					c.successfulFind(false, i);
				}
			}
			long finishTime = System.nanoTime();
			c.addTime(task.getNumber(), i, finishTime-time+errorPenalty);
		}
	}

	private void test5(int i) {

		Generic[] original = ds.getTask5Data(i);
		Generic[] copy = Arrays.copyOf(original, original.length);
		int index= c.sortAndSearch(copy, ds.getTask5Key(i));
		boolean pass = ds.checkTest5(c, copy, index, i);
		afterEffects(pass);

	}

	private void test4(int i) {

		int[][] original = ds.getTask4Data(i);
		int[][] copy = copyOf(original);
		double median = c.sortMultiDim(copy);
		boolean pass = ds.checkTest4(c, copy, median, i);
		afterEffects(pass);

	}
	
	private static int[][] copyOf(int[][] orig){
		int[][] copy = new int[orig.length][orig[0].length];
		for(int row = 0; row < copy.length; row++){
			for(int col = 0; col < copy[row].length; col++){
				copy[row][col] = orig[row][col];
			}
		}
		return copy;
	}

	private void test3(int i) {

		int[] original = ds.getTask3Data(i);
		int[] copy = Arrays.copyOf(original, original.length);
		double median = c.mostlySortAndGetMedian(copy);
		boolean pass = ds.checkTest3(c, copy, median, i);
		afterEffects(pass);

	}

	private void afterEffects(boolean pass){
		if (!pass && pair.getWinner() == null){
			pause(1.0);
		}else if(pass && pair.getWinner() == null){
			pair.damageOpponentOf(c);
		}
	}
	
	private void test2(int i) {

		String[] original = ds.getTask2Data(i);
		String[] copy = Arrays.copyOf(original, original.length);
		int index = c.sortAndGetResultingIndexOf(copy, ds.getTask2Key(i));
		boolean pass = ds.checkTest2(c, copy, index, i);
		afterEffects(pass);

	}

	private void test1(int i) {

		int[] original = ds.getTask1Data(i);
		int[] copy = Arrays.copyOf(original, original.length);
		double median = c.sortAndGetMedian(copy);
		boolean pass = ds.checkTest1(c, copy, median, i);
		afterEffects(pass);

	}

	private void pause(double i) {
		try {
			Thread.sleep((int)(1000*i));
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
