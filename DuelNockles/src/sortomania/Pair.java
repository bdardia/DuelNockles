package sortomania;

public final class Pair {

	private Contestant c1;
	private Contestant c2;
	private Contestant winner;
	
	
	public Pair(Contestant c1, Contestant c2) {
		this.c1 = c1;
		this.c2 = c2;
	}


	public Contestant getWinner() {
		return winner;
	}


	public synchronized void setWinner(Contestant winner) {
		if(Runner.screen.isKeepRunning()){
			setWinnerBasedOnSpeed(winner);
		}else{
			setWinnerBasedOnHP();
		}

	}


	private void setWinnerBasedOnHP() {
		boolean finish = this.winner != null; 
		if(c1.getHP() > c2.getHP()){
			winner = c1;
			c1.markVictorious(true);
			c2.markVictorious(false);
		}else{
			winner = c2;
			c2.markVictorious(true);
			c1.markVictorious(false);
		}
		if (finish) Runner.screen.notifyFinish();
		
	}


	private void setWinnerBasedOnSpeed(Contestant winner) {
		if(this.winner == null){
			this.winner = winner;
			winner.markVictorious(true);
			System.out.println("Winner declared.");
		}else{
			System.out.println("Winner already declared.");
			winner.markVictorious(false);
			Runner.screen.notifyFinish();
		}
	}


	public void damageOpponentOf(Contestant c) {
		Contestant target = c == c1 ? c2 : c1;
		target.damage(100.0/(DataSets.tasks * DataSets.startingReps*Runner.screen.getRound()));
	}
	
	

}
