import model.*;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.sql.*;
import java.util.*;

public class Main {
    public static void main(String[] args) throws SQLException, IOException {
        if(new DbConfig().dbConfigThen()){
            System.out.println("数据库连接成功！");
        }
        else{
            System.out.println("数据库连接失败… ");
        }

        Scanner sc=new Scanner(System.in);

        String TableName;

        while(true){
            System.out.print("请输入指令（exit退出）：");
            String order=sc.next();
            if(order.equals("exit")) break;

            //删除
            if(order.equals("delete")){
                Delete dl=new Delete();

                String mode=sc.next();
                if(mode.equals("data")){
                    System.out.print("请输入表格名、字段名：");
                    TableName=sc.next();
                    String key=sc.next();
                    System.out.print("请输入数据：");
                    sc.nextLine();
                    String Word=sc.nextLine();
                    if(dl.deleteData(TableName,key,Word)){
                        System.out.println("删除成功");
                    }
                    else{
                        System.out.println("删除失败");
                    }
                }
                else if(mode.equals("table")){
                    System.out.print("请输入表格名：");
                    String TableNames=sc.next();
                    if(dl.deleteTable(TableNames)){
                        System.out.println("删除成功");
                    }
                    else{
                        System.out.println("删除失败");
                    }
                }
            }

            //创建
            else if(order.equals("create")){
                System.out.print("文件是否自带字段（yes,no,WhatIsFile）：");
                Create ct=new Create();
                String Mode=sc.next();

                if(Mode.equals("yes")){
                    System.out.print("请输入文件名和表格名：");
                    String FileName=sc.next();
                    TableName=sc.next();
                    if(ct.createTable(FileName,TableName)){
                        System.out.println("创建成功！");
                    }
                    else{
                        System.out.println("创建失败！");
                    }
                }
                else if(Mode.equals("no")){
                    System.out.print("请输入文件名和表格名：");
                    String FileName=sc.next();
                    TableName=sc.next();
                    System.out.print("请输入字段名称（空格分隔）：");
                    sc.nextLine();
                    String keys=sc.nextLine();
                    if(ct.createTable(FileName,TableName,keys)){
                        System.out.println("创建成功！");
                    }
                    else{
                        System.out.println("创建失败！");
                    }
                }
                else if(Mode.equals("WhatIsFile")){
                    System.out.print("请输入表格名：");
                    TableName=sc.next();
                    System.out.print("请输入字段名称（空格分隔）：");
                    sc.nextLine();
                    String keys=sc.nextLine();
                    if(ct.createOnlyTable(TableName,keys)){
                        System.out.println("创建成功！");
                    }
                    else{
                        System.out.println("创建失败！");
                    }
                }
                else{
                    System.out.println("Mode Wrong");
                }
            }

            //插入
            else if(order.equals("insert")){
                Insert ist=new Insert();
                String mode;
                mode=sc.next();
                if(mode.equals("data")){//手动
                    System.out.print("选择表格：");

                    TableName=sc.next();
                    System.out.print("输入字段：");
                    sc.nextLine();
                    String keys=sc.nextLine();

                    if(ist.insertData(TableName,keys)){
                        System.out.println("插入成功！");
                    }
                    else{
                        System.out.println("插入失败！");
                    }
                }
                else if(mode.equals("file")){//文件
                    System.out.print("选择文件及表格：");

                    String fileName=sc.next();
                    TableName=sc.next();

                    Path path= Paths.get("project_dataset\\"+fileName);
                    List<String> lines= Files.readAllLines(path);
                    Insert is=new Insert();
                    if(is.insertFile(TableName,lines)){
                        System.out.println("导入完毕");
                    }
                    else{
                        System.out.println("导入失败");
                    }
                }
                else{
                    System.out.println("wrong");
                }
            }

            //查询
            else if(order.equals("select")){
                System.out.print("选择模式（all,some,one)：");
                String mode=sc.next();

                if(mode.equals("all")){
                    System.out.print("输入表格名：");
                    TableName=sc.next();
                    if(new Select().selectAllData(TableName)){
                        System.out.println("查询成功！");
                        String Page;

                        while(true){
                            System.out.print("输入要查询的页码：");
                            Page=sc.next();
                            if(Page.equals("exit")) {
                                System.out.println("退出");
                                break;
                            }
                            if(new Select().selectAllData(TableName,Page)){
                                System.out.println("查询成功！");
                            }else{
                                System.out.println("查询失败！");
                            }
                        }
                    }
                    else{
                        System.out.println("查询失败！");
                    }

                }

                else if(mode.equals("some")){
                    System.out.print("输入表格名、所查字段名、数据：");
                    TableName=sc.next();
                    String column=sc.next();
                    String data=sc.next();

                    if(new Select().selectData(TableName,column,data)){
                        System.out.println("查询成功！");
                        String Page;
                        while(true){
                            System.out.print("输入要查询的页码：");
                            Page=sc.next();
                            if(Page.equals("exit")) {
                                System.out.println("退出");
                                break;
                            }
                            if(new Select().selectData(TableName,column,data,Page)){
                                System.out.println("查询成功！");
                            }else{
                                System.out.println("查询失败！");
                            }
                        }
                    }
                    else{
                        System.out.println("查询失败！");
                    }


                }
                else if(mode.equals("one")){

                    System.out.print("输入表格名、所查字段名：");
                    TableName=sc.next();
                    String columnName=sc.next();
                    sc.nextLine();
                    System.out.print("输入数据：");
                    String Data=sc.nextLine();
                    if(new Select().selectOneData(TableName,columnName,Data)){
                        System.out.println("查询成功！");
                        String Page;
                        while(true){
                            System.out.print("输入要查询的页码：");
                            Page=sc.next();
                            if(Page.equals("exit")) {
                                System.out.println("退出");
                                break;
                            }
                            if(new Select().selectOneData(TableName,columnName,Data,Page)){
                                System.out.println("查询成功！");
                            }else{
                                System.out.println("查询失败！");
                            }
                        }
                    }
                    else{
                        System.out.println("查询失败！");
                    }
                }
                else{
                    System.out.println("模式错误");
                }
            }

            //更新
            else if(order.equals("update")){
                Update ud=new Update();
                System.out.print("请输入表格名、需改字段、新数据、特有字段、特有数据：");
                TableName=sc.next();
                String column=sc.next();
                String newData=sc.next();
                String addColumn=sc.next();
                String condition=sc.next();
                if(ud.updateData(TableName,column,newData,addColumn,condition)){
                    System.out.println("更新成功！");
                }
                else{
                    System.out.println("更新失败！");
                }
            }

            //导出
            else if(order.equals("export")){

                System.out.print("选择模式（all,some）：");
                String mode=sc.next();
                if(mode.equals("all")){
                    System.out.print("表格名、文件路径及名称：");
                    Export ep=new Export();
                    TableName=sc.next();
                    String fileName=sc.next();

                    if(ep.exportTXT(TableName,fileName)){
                        System.out.println("导出成功！");
                    }
                    else{
                        System.out.println("导出失败！");
                    }
                }
                else if(mode.equals("some")){
                    System.out.print("表格名、文件路径及名称：");
                    Export ep=new Export();
                    TableName=sc.next();
                    String fileName=sc.next();

                    System.out.print("导出页码：");
                    String page=sc.next();
                    String[] pages=page.split(",");
                    if(ep.exportPage(TableName,fileName,pages)){
                        System.out.println("导出成功！");
                    }
                    else{
                        System.out.println("导出失败！");
                    }
                }
            }

            else{
                System.out.println("指令输入错误！");
            }
        }
    }
}