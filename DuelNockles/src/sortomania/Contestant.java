package sortomania;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import javax.imageio.ImageIO;

import ui.Component;

public abstract class Contestant extends Component implements Runnable{

	private String string;
	private int contestantNumber;
	private String title;
	private String status;
	private int currentTask;
	private int points;
	private double currentScore;
	private boolean wonLastRound;
	private int correctSorts;
	private int totalSorts;
	private int correctMedians;
	private int totalMedians;
	private long[][] recordedTimes;
	private double[] bestAverages;
	private long currentTaskTime;
	private boolean showTime;
	private String recentError;
	private long errorTime;

	private boolean attacking;
	private boolean hit;
	private int topMargin = 64;
	private int sideMargin = 0;
	private int textMargin = 80;
	public static final int REFRESH_RATE = 20;
	private boolean running;
	private ArrayList<BufferedImage> frame; //the images that can be displayed
	private ArrayList<Integer> times; //the time each image is displayed
	private ArrayList<BufferedImage> attackFrame; //the images that can be displayed
	private ArrayList<Integer> attackTimes; //the time each image is displayed
	private ArrayList<BufferedImage> hitFrame; //the images that can be displayed
	private ArrayList<Integer> hitTimes; //the time each image is displayed

	private long displayTime; //the time when the last image switched
	private int currentFrame; //the frame that is currently being displayed


	public static final String KEN = "resources/ken-sprite-sheet.png";
	public static final String RYU = "resources/ryu-sprite-sheet.png";
	public static final String CHUN_LI = "resources/chun-li-sprite-sheet.png";
	public static final String DEE_JAY = "resources/dee-jay-sprite-sheet.png";
	public static final String BLANKA = "resources/blanka-sprite-sheet.png";
	public static final String E_HONDA = "resources/e-honda-sprite-sheet.png";
	public static final String FEI_LONG = "resources/fei-long-sprite-sheet.png";
	public static final String CAMMY = "resources/cammy-sprite-sheet.png";

	public Contestant() {
		super(0, 0, 310, 170);
		this.string = this.getClass().toString().replaceAll("class sortomania.contestants.", "");
		currentTaskTime = System.currentTimeMillis();
		recordedTimes = new long[5][15];
		bestAverages = new double[5];
		currentScore = 100;
		wonLastRound = true;
		showTime = false;
		status = "";
		frame = new ArrayList<BufferedImage>();
		attackFrame = new ArrayList<BufferedImage>();
		hitFrame = new ArrayList<BufferedImage>();
		times = new ArrayList<Integer>();
		attackTimes = new ArrayList<Integer>();
		hitTimes = new ArrayList<Integer>();
		currentFrame = 0;
		addFrames(getSpriteName());
	}

	public  final void addFrame(ArrayList<BufferedImage> set, ArrayList<Integer> timeset, BufferedImage image, Integer time){
		set.add(image);
		timeset.add(time);
	}

	/**
	 * returns the team color
	 * To return a color, retrieve them statically using Color.blue/red/green/etc
	 * Or customize a color using RGB values, for example:
	 * return new Color(255,200,200);
	 * @return
	 */
	public abstract Color getColor();


	private void addFrames(String spriteName) {
		int[] values = new int[15];
		if(spriteName.equals(KEN)){
			int[] temp = {7,15,49,90,4,5,130,49,85,3,211,774,56,85,4};
			values = temp;
		}else if(spriteName.equals(RYU)){
			int[] temp = {7,15,49,90,4,5,130,49,85,3,211,774,56,85,4};
			values = temp;
		}else if(spriteName.equals(CHUN_LI)){
			int[] temp = {4,30,53,82,4,
					435,141,69,82,3,
					2,850,56,82,2};
			values = temp;
		}else if(spriteName.equals(DEE_JAY)){
			int[] temp = {8,32,56,92,4,
					6,152,55,92,3,
					887,1123,51,92,3};
			values = temp;
		}else if(spriteName.equals(BLANKA)){
			int[] temp = {8,33,70,90,4,
					0,152,81,90,3,
					4,1167,66,90,3};
			values = temp;
		}else if(spriteName.equals(E_HONDA)){
			int[] temp = {5,45,73,87,4,
					6,152,75,90,3,
					6,910,66,90,3};
			values = temp;
		}else if(spriteName.equals(FEI_LONG)){
			int[] temp = {3,21,58,82,8,
					6,131,55,85,3,
					640,770,60,77,3};
			values = temp;
		}else if(spriteName.equals(CAMMY)){
			int[] temp = {3,21,53,82,5,
					0,131,63,77,3,
					405,893,55,85,4};
			values = temp;
		}else if(spriteName.equals("resources/penguinmaple.png")) {
			int[] temp = {0,0,118,76,8,
					  0,161,126,84,4,
					 0,449,174,98,2};
			values = temp;
		}
		
		try {
			BufferedImage originalImgage = ImageIO.read(new File(spriteName));
			addSequence(originalImgage,frame, times, values[0],values[1],values[2],values[3],values[4]);
			addSequence(originalImgage,attackFrame, attackTimes, values[5],values[6],values[7],values[8],values[9]);
			addSequence(originalImgage,hitFrame, hitTimes, values[10],values[11],values[12],values[13],values[14]);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}



	private void addSequence(BufferedImage originalImgage, ArrayList<BufferedImage> addTo, ArrayList<Integer> times, int x, int y, int w, int h,
			int n) {
		for(int i = 0; i < n; i++){
			addFrame(addTo,times,originalImgage.getSubimage(x+w*i, y, w, h), 200);
		}
	}

	/**
	 * 
	 * @return on of the Sprite sheet image addresses. Available Sprites are from Street Fighter 2.
	 *return KEN;
	 *return RYU;
	 *return CHUN_LI;
	 *return DEE_JAY;
	 *return BLANKA;
	 *return CAMMY;
	 *return E_HONDA;
	 *return FEI_LONG;
	 */
	public abstract String getSpriteName();

	public final void setNumber(int i){
		contestantNumber = i;
		title = "Contestant "+i;
		update();
	}

	public String toString(){
		return string;
	}

	/**
	 * TASK 1
	 * sorts a completely randomized array and returns the median
	 * @param random
	 * @return
	 */
	public abstract double sortAndGetMedian(int[] random);

	/**
	 * TASK 2
	 * sorts a completely randomized array of random Strings and returns the index of
	 * the given String in the resulting sorted array
	 * @param random
	 * @return
	 */
	public abstract int sortAndGetResultingIndexOf(String[] strings, String toFind);


	/**
	 * TASK 3
	 * sorts a mostly sorted array and returns the median
	 * @param random
	 * @return
	 */
	public abstract double mostlySortAndGetMedian(int[] mostlySorted);

	/**
	 * TASK 4
	 * sorts each array in a multi-dimensional array
	 * then returns the median of the medians
	 * @param random
	 * @return
	 */
	public abstract double sortMultiDim(int[][] grid);

	/**
	 * TASK 5
	 * TASK 2 (same directions)
	 * @param random
	 * @return
	 */
	public abstract int sortAndSearch(Comparable[] arr, Comparable toFind);

	@Override
	public void update(Graphics2D g) {
		g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		g.setColor(Color.WHITE);
		g.fillRect(0, 0, getWidth(), getHeight());
		drawImage(g);
		g.setColor(getColor());
		g.fillRect(5, 5, 35, 35);
		g.setFont(new Font("Futura", Font.PLAIN, 10));
		g.setColor(Color.BLACK);
		g.setStroke(new BasicStroke(2));
		g.drawRect(5, 5, 35, 35);
		g.setStroke(new BasicStroke(1));
		String time = showTime ? ", Current Task: "+((int)(100*(System.currentTimeMillis()-currentTaskTime))/100000.0):"";
				
		g.drawString(string+", Score: "+points+time, 45, 20);
		g.drawString(status, 45, 35);
		if(currentScore <25){
			g.setColor(Color.red);
		}else if(currentScore < 50){
			g.setColor(Color.yellow);
		}else if(currentScore < 70){
			g.setColor(new Color(190,255,200));
		}else if(currentScore < 90){
			g.setColor(Color.green);
		}else {
			g.setColor(Color.green);
		}
		g.fillRect(0, 45, (int)((currentScore/100.0)*(getWidth()-2)), 18);
		
		long timeSinceError = System.currentTimeMillis()-errorTime;
		if(timeSinceError < 1000){
			System.out.println("Drawing error text");
			g.setColor(new Color(0,0,0,(int)(255.0*(1-timeSinceError/1000.0))));
			g.drawString(recentError, 5, 58);
		}
		
		g.setColor(Color.BLACK);
		g.drawRect(0, 45, getWidth()-2, 18);
		g.drawString("Correct Sorts: "+correctSorts+"/"+totalSorts, textMargin, topMargin+15);
		g.drawString("Correct Medians: "+correctMedians+"/"+totalMedians, textMargin, topMargin+25);
		for(int index = 0; index < bestAverages.length; index++){
			g.drawString("Test "+(index+1)+" average of top 15: "+bestAverages[index]+"ms", textMargin, topMargin+35+(12*(index+1)));
		}

	}
	

	public final void move(int x, int y){
		double xDif = x - getX();
		double yDif = y - getY();
		int origX = getX();
		int origY = getY();
		Thread movement = new Thread(new Runnable() {

			@Override
			public void run() {
				int count = 0;
				double limit = 20;
				while(count <=limit){
					Contestant.this.setX((int)(origX+(xDif/limit)*count));
					Contestant.this.setY((int)(origY+(yDif/limit)*count));
					try {
						Thread.sleep(75);
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					count++;
				}
			}
		});
		movement.start();
	}

	public final void run(){

		running = true;
		while(running){

			try {
				Thread.sleep(REFRESH_RATE);
				update();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		}
	}

//	public final void move(int x, int y){
//		double xDif = x - getX();
//		double yDif = y - getY();
//		int origX = getX();
//		int origY = getY();
//		Thread movement = new Thread(new Runnable() {
//
//			@Override
//			public void run() {
//				int count = 0;
//				double limit = 20;
//				while(count <=limit){
//					Contestant.this.setX((int)(origX+(xDif/limit)*count));
//					Contestant.this.setY((int)(origY+(yDif/limit)*count));
//					try {
//						Thread.sleep(75);
//					} catch (InterruptedException e) {
//						// TODO Auto-generated catch block
//						e.printStackTrace();
//					}
//					count++;
//				}
//			}
//		});
//		movement.start();
//	}

	public final void drawImage(Graphics2D g) {
		long currentTime = System.currentTimeMillis();//gets time now

		ArrayList<BufferedImage> frame;
		ArrayList<Integer> times;
		if(attacking && !hit){
			frame = attackFrame;
			times = attackTimes;
			if(currentFrame == attackFrame.size()-1){
				frame = this.frame;
				times = this.times;
				currentFrame = frame.size() -1;
				attacking = false;
			}
		}else if(hit){
			frame = hitFrame;
			times = hitTimes;
			if(currentFrame == hitFrame.size()-1){
				frame = this.frame;
				times = this.times;
				currentFrame = frame.size() -1;
				hit = false;
			}
		}else{
			frame = this.frame;
			times = this.times;
		}

		
		//finally after all adjustments have been made, make sure index is not out of bounds
		if(currentFrame >= frame.size()){
			currentFrame = 0;
		}
		//check if it's time to change the frame
		//and make sure that there are images in the frame list
		if(frame != null && frame.size() > 0 && frame.size() == times.size() && currentTime - displayTime > times.get(currentFrame)){
			displayTime = currentTime;

			//increase the currentFrameIndex but don't exceed size()
			currentFrame = (currentFrame+1)%frame.size();

			//clear the previous image
			BufferedImage newFrame = frame.get(currentFrame);
			g.drawImage(newFrame,sideMargin,topMargin, null);
			//			g.drawImage(newFrame, sideMargin,topMargin,getWidth()-2*sideMargin,getHeight()-topMargin,0,0,newFrame.getWidth(),newFrame.getHeight(),null);
		}else{
			BufferedImage newFrame = frame.get(currentFrame);
			g.drawImage(newFrame,sideMargin,topMargin, null);
			//			g.drawImage(newFrame, sideMargin,topMargin,getWidth()-2*sideMargin,getHeight()-topMargin,0,0,newFrame.getWidth(),newFrame.getHeight(),null);
		}
	}

	/**
	 * Performs a show demonstrating the Contestant has started the next task
	 * @param i
	 */
	public final void beginTask(int i) {
		showTime = true;
		currentTask = i;
		currentTaskTime = System.currentTimeMillis();
//		status = "Task "+i+"... ";
	}

	public final void successfulSort(boolean b, int trial) {
		String report = (b)?"sorted "+trial+"! ":"failed "+trial+"! ";
		status = "Task "+currentTask+"... "+report;
		totalSorts += 1;
		if(b){
			correctSorts+=1;
//			currentScore+=.1;
		}else{
			currentScore-=.1;
		}
		update();
	}

	public final void successfulFind(boolean b, int trial) {
		String report = (b)?"found "+trial+"! ":"missed "+trial+"! ";
		status += report;
		totalMedians += 1;
		if(b){
			correctMedians += 1;
			attacking = true;

		}else{
			currentScore-=.1;
		}
		update();
	}

	public final boolean wonLastRound(){
		return wonLastRound; 
	}

	public final void markVictorious(boolean b) {
		wonLastRound = b;
		showTime = false;
		String report = (b)?"Victory! ":"Defeat! ";
		if (b){
			points += 1;	
		}

		status = report;
		update();
	}

	public final void penalize(String string, int time) {
		hit = true;
		recentError = string;
		errorTime = System.currentTimeMillis();
		if(time > 0){
			try {
				Thread.sleep(time);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	public final int getHP() {
		return (int)(currentScore);
	}

	public final void addTime(int test, int trial, long l) {
		int i = 0;
		boolean addedNew = false;
		while(i < recordedTimes[test-1].length){
			if(recordedTimes[test-1][i] == 0 || l < recordedTimes[test-1][i]){
				recordedTimes[test-1][i] = l;
				addedNew = true;
				break;
			}
			i++;
		}
		if (addedNew){
			double sum = 0;
			for(int j = 0; j < recordedTimes[test-1].length; j++){
				sum += recordedTimes[test-1][j];
			}
			bestAverages[test-1] = (int)(10000*(sum / (recordedTimes[test-1].length*1000000)))/10000.0;
		}
	}
	
	public double getScore(){
		return correctSorts/(double)totalSorts;
	}
	
	
	public int getPoints(){
		return points;
	}
	
	
	public final void damage(double d) {
		currentScore -= d;
		
	}
	
	public final void setHP(double i) {
		currentScore = i;
	}

	public int getCorrectSorts() {
		return correctSorts;
	}

	public int getTotalSorts() {
		return totalSorts;
	}

	public int getCorrectMedians() {
		return correctMedians;
	}

	public int getTotalMedians() {
		return totalMedians;
	}

	

}
