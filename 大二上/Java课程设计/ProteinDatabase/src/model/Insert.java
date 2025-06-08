package model;

import jdk.jshell.StatementSnippet;

import java.sql.*;
import java.util.List;

public class Insert {

    Connection connection=new DbConfig().dbConfig();

    public Insert() throws SQLException {
    }

    public boolean insertData(String TableName,String keys) throws SQLException {
        Statement statement=connection.createStatement();
        String sql="select * from "+TableName;
        ResultSet rs=statement.executeQuery(sql);
        ResultSetMetaData md=rs.getMetaData();

        String counts="";
        for(int i=0;i<md.getColumnCount();i++){
            if(i==0) counts+="?";
            else{
                counts+=",?";
            }
        }

        String a[]=keys.split(" ");
        sql="insert into "+TableName+" values("+counts+")";
        PreparedStatement pst=connection.prepareStatement(sql);
        for(int i=1;i<=md.getColumnCount();i++){
            pst.setObject(i,a[i-1]);
        }
        int i=pst.executeUpdate();
        return i==1;
    }

    public boolean insertFile(String TableName,List<String> lines) throws SQLException {

        try{
            long startTime = System.currentTimeMillis();
            connection.setAutoCommit(false);

            Statement statement=connection.createStatement();
            String sql="select * from "+TableName;
            ResultSet rs=statement.executeQuery(sql);
            ResultSetMetaData md=rs.getMetaData();

            String counts="";
            for(int i=0;i<md.getColumnCount();i++){
                if(i==0) counts+="?";
                else{
                    counts+=",?";
                }
            }
            sql="insert into "+TableName+" values("+counts+")";
            PreparedStatement pst=connection.prepareStatement(sql);
            long count=0;
            for(String line:lines){
                count++;
                if(count==1) continue;
                String a[]=line.split("\t");

                for(int i=1;i<=md.getColumnCount();i++){
                    pst.setObject(i,a[i-1]);
                }
                pst.addBatch();
                if(count%2000==0){
                    pst.executeBatch();
                    connection.commit();
                    pst.clearBatch();
                    System.out.println(count+"条数据");
                }
                //pst.executeUpdate();
            }
            count--;
            System.out.println("共"+count+"条数据");
            pst.executeBatch();
            connection.commit();
            pst.clearBatch();
            long endTime = System.currentTimeMillis();

            System.out.println(endTime-startTime+"ms");
            return true;
        } catch (Exception e) {
            return false;
        }
    }
}
