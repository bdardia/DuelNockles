package sortomania;

import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

import sortomania.contestants.NabeelAmanatSort;

public class CSV {

    //Delimiter used in CSV file

private static final String COMMA_DELIMITER = ",";

private static final String NEW_LINE_SEPARATOR = "\n";

	private ArrayList<Contestant> students; 
	
	public static void main(String[] args) {
		ArrayList<Contestant> participants = new ArrayList<Contestant>();
		participants.add(new NabeelAmanatSort());
		CSV csv = new CSV(participants);
		csv.writeCsvFile("Testing.csv");
		
	}
	
	public CSV(ArrayList<Contestant> participants) {
		this.students = new ArrayList<Contestant>();
		students.addAll(participants);
		
		
	}
	
	public CSV(ArrayList<Contestant> participants, ArrayList<Contestant> runnersUp) {
		this.students = new ArrayList<Contestant>();
		students.addAll(participants);
		students.addAll(runnersUp);
		
	}

	public  void writeCsvFile(String fileName) {

		        FileWriter fileWriter = null;

		        try {
	
		            fileWriter = new FileWriter(fileName);

		            fileWriter.append("Class name,Rounds Won,Score,Correct Sorts,Attempted Sorts,Correct Objectives,Attempted Objectives");

		            fileWriter.append(NEW_LINE_SEPARATOR);

		            for (Contestant student : students) {
		                fileWriter.append(String.valueOf(student.getClass().toString().replaceAll("class sortomania.contestants.", "")));
		                fileWriter.append(COMMA_DELIMITER);
		                fileWriter.append(student.getPoints()+"");
		                fileWriter.append(COMMA_DELIMITER);
		                fileWriter.append(student.getScore()+"");
		                fileWriter.append(COMMA_DELIMITER);
		                fileWriter.append(student.getCorrectSorts()+"");
		                fileWriter.append(COMMA_DELIMITER);
		                fileWriter.append(student.getTotalSorts()+"");
		                fileWriter.append(COMMA_DELIMITER);
		                fileWriter.append(student.getCorrectMedians()+"");
		                fileWriter.append(COMMA_DELIMITER);
		                fileWriter.append(student.getTotalMedians()+"");
		                fileWriter.append(NEW_LINE_SEPARATOR);
		            }

		            System.out.println("CSV file was created successfully !!!");

		        } catch (Exception e) {
		            System.out.println("Error in CsvFileWriter !!!");
		
		            e.printStackTrace();
		
		        } finally {

		            try {
		
		                fileWriter.flush();
		
		                fileWriter.close();
		
		            } catch (IOException e) {
		
		                System.out.println("Error while flushing/closing fileWriter !!!");
		
		                e.printStackTrace();
		
		            }
		        }
		    }

	
}
