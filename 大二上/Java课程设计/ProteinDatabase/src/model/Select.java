package model;

import java.sql.*;

public class Select {
    Connection connection=new DbConfig().dbConfig();

    public Select() throws SQLException {
    }

    public boolean selectTable(String TableName) throws SQLException {
        String sql="select * from "+TableName;
        Statement statement=connection.createStatement();

        try{
            ResultSet rs=statement.executeQuery(sql);
            return true;
        } catch (SQLException e) {
            return false;
        }
    }

    public long selectNum(String TableName) throws SQLException {
        String sql="select * from "+TableName;
        Statement statement=connection.createStatement();

        try{
            ResultSet rs=statement.executeQuery(sql);
            long count=0;
            while (rs.next()){
                count++;
            }
            return count;
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    public boolean selectAllData(String TableName) throws SQLException {

        String sql="select * from "+TableName+ " limit (1-1)*100,100";
        String sql_All="select * from "+TableName;
        Statement statement=connection.createStatement();

        try {
            ResultSet rs=statement.executeQuery(sql);
            ResultSetMetaData md=rs.getMetaData();

            for (int i=0;i<md.getColumnCount();i++){
                System.out.print(String.format("| %-25s ",md.getColumnName(i+1)));
            }
            System.out.println();
            long count=selectNum(TableName);
            while(rs.next()){
                for(int i=0;i<md.getColumnCount();i++){
                    int endNum=Math.min(rs.getString(i+1).length(),22);
                    if(rs.getString(i+1).length()>=23)
                        System.out.print(String.format("| %-22s... ",rs.getString(i+1).substring(0,endNum)));
                    else
                        System.out.print(String.format("| %-25s ",rs.getString(i+1).substring(0,endNum)));
                }
                System.out.println();
            }
            System.out.println("共"+count+"条，当前第1页");
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    public boolean selectAllData(String TableName,String Page) throws SQLException {

        String sql="select * from "+TableName+ " limit ("+Page+"-1)*100+1,100";
        Statement statement=connection.createStatement();

        try {
            ResultSet rs=statement.executeQuery(sql);
            ResultSetMetaData md=rs.getMetaData();
            for (int i=0;i<md.getColumnCount();i++){
                System.out.print(String.format("| %-25s ",md.getColumnName(i+1)));
            }
            System.out.println();
            long count=selectNum(TableName);
            while(rs.next()){
                for(int i=0;i<md.getColumnCount();i++){
                    int endNum=Math.min(rs.getString(i+1).length(),22);
                    if(rs.getString(i+1).length()>=23)
                        System.out.print(String.format("| %-22s... ",rs.getString(i+1).substring(0,endNum)));
                    else
                        System.out.print(String.format("| %-25s ",rs.getString(i+1).substring(0,endNum)));
                }
                System.out.println();
            }
            System.out.println("共"+count+"条，当前第"+Page+"页");
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    public boolean selectData(String TableName,String ColumnName,String Data,String Page) throws SQLException {

        String sql="select * from "+TableName+" where "+ColumnName+" like '%"+Data+"%' limit ("+Page+"-1)*100,100";
        String sqll="select * from "+TableName+" where "+ColumnName+" like '%"+Data+"%'";
        Statement statement=connection.createStatement();

        try {
            ResultSet rs=statement.executeQuery(sql);
            ResultSetMetaData md=rs.getMetaData();


            for (int i=0;i<md.getColumnCount();i++){
                System.out.print(String.format("| %-25s ",md.getColumnName(i+1)));
            }
            System.out.println();

            while(rs.next()){
                for(int i=0;i<md.getColumnCount();i++){
                    int endNum=Math.min(rs.getString(i+1).length(),22);
                    if(rs.getString(i+1).length()>=23)
                        System.out.print(String.format("| %-22s... ",rs.getString(i+1).substring(0,endNum)));
                    else
                        System.out.print(String.format("| %-25s ",rs.getString(i+1).substring(0,endNum)));
                }
                System.out.println();
            }
            long count=0;
            rs=statement.executeQuery(sqll);
            while(rs.next()){
                count++;
            }
            System.out.println("共"+count+"条，当前第"+Page+"页");

            return true;
        } catch (Exception e) {
            return false;
        }
    }

    //
    public boolean selectData(String TableName,String ColumnName,String Data) throws SQLException {

        String sql="select * from "+TableName+" where "+ColumnName+" like '%"+Data+"%' limit (1-1)*100,100";
        String sqll="select * from "+TableName+" where "+ColumnName+" like '%"+Data+"%'";
        Statement statement=connection.createStatement();

        try {
            ResultSet rs=statement.executeQuery(sql);
            ResultSetMetaData md=rs.getMetaData();
            for (int i=0;i<md.getColumnCount();i++){
                System.out.print(String.format("| %-25s ",md.getColumnName(i+1)));
            }
            System.out.println();

            long count=0;
            while(rs.next()){
                for(int i=0;i<md.getColumnCount();i++){
                    int endNum=Math.min(rs.getString(i+1).length(),22);
                    if(rs.getString(i+1).length()>=23)
                        System.out.print(String.format("| %-22s... ",rs.getString(i+1).substring(0,endNum)));
                    else
                        System.out.print(String.format("| %-25s ",rs.getString(i+1).substring(0,endNum)));
                }
                System.out.println();
            }

            rs=statement.executeQuery(sqll);
            while(rs.next()){
                count++;
            }
            System.out.println("共"+count+"条，当前第1页");
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    public boolean selectOneData(String TableName,String columnName,String Data) throws SQLException {
        String sql="select * from "+TableName+" where "+columnName+" = '"+Data+"' limit (1-1)*100,100";
        String sqll="select * from "+TableName+" where "+columnName+" = '"+Data+"'";
        Statement statement=connection.createStatement();
        System.out.println(Data);

        try{
            ResultSet rs=statement.executeQuery(sql);
            ResultSetMetaData md=rs.getMetaData();
            int cnt=0;
            while (rs.next()){
                cnt++;
                System.out.println("第"+cnt+"条");
                for(int i=0;i<md.getColumnCount();i++){
                    System.out.println(String.format("%-20s",md.getColumnName(i+1))+":"+rs.getString(i+1));
                }
                System.out.println();
            }
            long count=0;
            rs=statement.executeQuery(sqll);
            while(rs.next()){
                count++;
            }
            System.out.println("共"+count+"条，当前第1页");

            return true;
        } catch (SQLException e) {
            return false;
        }
    }

    public boolean selectOneData(String TableName,String columnName,String Data,String Page) throws SQLException {
        String sql="select * from "+TableName+" where "+columnName+" = '"+Data+"' limit ("+Page+"-1)*100,100";
        String sqll="select * from "+TableName+" where "+columnName+" = '"+Data+"'";

        Statement statement=connection.createStatement();

        try{
            ResultSet rs=statement.executeQuery(sql);
            ResultSetMetaData md=rs.getMetaData();
            int cnt=0;
            while (rs.next()){
                cnt++;
                System.out.println("第"+cnt+"条");
                for(int i=0;i<md.getColumnCount();i++){
                    System.out.println(String.format("%-20s",md.getColumnName(i+1))+":"+rs.getString(i+1));
                }
                System.out.println();
            }
            long count=0;
            rs=statement.executeQuery(sqll);
            while(rs.next()){
                count++;
            }
            System.out.println("共"+count+"条，当前第"+Page+"页");

            return true;
        } catch (SQLException e) {
            return false;
        }
    }
}
