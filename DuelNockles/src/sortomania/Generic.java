package sortomania;

public class Generic implements Comparable<Generic> {
	
	int value = 0;
	
	public Generic()
	{
		value = (int)(Math.random() * 10000);
	}
	
	public int compareTo(Generic g)
	{
		return this.value - g.getValue();
	}
	
	public int getValue()
	{
		return this.value;
	}
}