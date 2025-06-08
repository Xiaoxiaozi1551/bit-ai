package model;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.List;

public class Create {
    Connection connection=new DbConfig().dbConfig();

    public Create() throws SQLException {
    }

    //存在表头
    public boolean createTable(String FileName,String TableName) throws IOException, SQLException {

        if(new Select().selectTable(TableName)){
            System.out.println("表存在");
        }

        else{
            BufferedReader br=new BufferedReader(new FileReader("project_dataset\\"+FileName));
            String text=br.readLine();
            String sqll="";
            sqll.replace(" ","");
            String []a=text.split("\t");
            for(int j=0;j<a.length-1;j++){
                if(a[j].equals("")||a[j].equals(" ")) {
                    a[j]="tempName";
                }
                if(a[j].equals("index")){
                    a[j]="idx";
                }
                a[j]+=" varchar(50), ";
                sqll+=a[j];
            }
            sqll=sqll+a[a.length-1]+" varchar(50)";
            String sql="create table if not exists "+TableName+"(" +
                    sqll+
                    ")";
            System.out.println(sql);
            Statement statement=connection.createStatement();
            statement.executeUpdate(sql);//创建

            //导入
            Path path= Paths.get("project_dataset\\"+FileName);
            List<String> lines= Files.readAllLines(path);
            Insert is=new Insert();
            if(is.insertFile(TableName,lines)){
                System.out.println("导入完毕");
            }

            if(new Select().selectTable(TableName)){
                return true;
            }
        }
        return false;
    }

    //不存在表头
    public boolean createTable(String FileName, String TableName, String keys) throws IOException, SQLException {

        if(new Select().selectTable(TableName)){
            System.out.println("表存在");
        }
        else{
            BufferedReader br=new BufferedReader(new FileReader("project_dataset\\"+FileName));
            String sqll="";
            String[] a=keys.split(" ");
            for(int j=0;j<a.length-1;j++){
                if(a[j].equals("")||a[j].equals(" ")) {
                    a[j]="idx";
                }
                if(a[j].equals("index")){
                    a[j]="idx";
                }
                a[j]+=" varchar(50), ";
                sqll+=a[j];
            }
            sqll=sqll+a[a.length-1]+" varchar(50)";
            String sql="create table "+TableName+"(" +
                    sqll+
                    ")";
            System.out.println(sql);
            Statement statement=connection.createStatement();
            statement.executeUpdate(sql);

            //导入
            Path path= Paths.get("project_dataset\\"+FileName);
            List<String> lines= Files.readAllLines(path);
            Insert is=new Insert();
            if(is.insertFile(TableName,lines)){
                System.out.println("导入完毕");
            }

            if(new Select().selectTable(TableName)){
                return true;
            }
        }
        return false;
    }

    //只创建表格
    public boolean createOnlyTable(String TableName, String keys) throws IOException, SQLException {

        if(new Select().selectTable(TableName)){
            System.out.println("表存在");
        }
        else{
            String sqll="";
            String[] a=keys.split(" ");
            for(int j=0;j<a.length-1;j++){
                if(a[j].equals("")||a[j].equals(" ")) {
                    a[j]="tempName";
                }
                if(a[j].equals("index")){
                    a[j]="idx";
                }
                a[j]+=" varchar(50), ";
                sqll+=a[j];
            }
            sqll=sqll+a[a.length-1]+" varchar(50)";
            String sql="create table "+TableName+"(" +
                    sqll+
                    ")";
            System.out.println(sql);
            Statement statement=connection.createStatement();
            statement.executeUpdate(sql);

            if(new Select().selectTable(TableName)){
                return true;
            }
        }
        return false;
    }
}
