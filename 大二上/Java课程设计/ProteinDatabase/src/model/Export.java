package model;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.sql.*;

public class Export {
    Connection connection=new DbConfig().dbConfig();

    public Export() throws SQLException {
    }

    public boolean exportTXT(String TableName,String fileName){
        try{
            File file=new File(fileName);
            if(!file.exists()){
                file.createNewFile();
            }

            FileWriter writer =new FileWriter(fileName);
            BufferedWriter out = new BufferedWriter(writer);
            String sql="select * from "+TableName;

            Statement statement=connection.createStatement();

            ResultSet rs=statement.executeQuery(sql);
            ResultSetMetaData md=rs.getMetaData();

            while (rs.next()){
                for(int i=0;i<md.getColumnCount();i++){
                    out.write(rs.getString(i+1)+"\t");
                }
                out.write("\n");
            }
            out.flush();
            out.close();

            return true;

        } catch (Exception e) {
            return false;
        }
    }


    public boolean exportPage(String TableName,String fileName,String[] pages){
        try{
            File file=new File(fileName);
            if(!file.exists()){
                file.createNewFile();
            }
            else{

            }

            FileWriter writer =new FileWriter(fileName,true);
            BufferedWriter out = new BufferedWriter(writer);
            String sql=null;
            for (String page:pages){
                sql="select * from "+TableName+" limit ("+page+"-1)*100,100";
                Statement statement=connection.createStatement();

                ResultSet rs=statement.executeQuery(sql);
                ResultSetMetaData md=rs.getMetaData();

                while (rs.next()){
                    for(int i=0;i<md.getColumnCount();i++){
                        out.write(rs.getString(i+1)+"\t");
                    }
                    out.write("\n");
                }

            }
            out.flush();
            out.close();
            return true;

        } catch (Exception e) {
            return false;
        }
    }
}
