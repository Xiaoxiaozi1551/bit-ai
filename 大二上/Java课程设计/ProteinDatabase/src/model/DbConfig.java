package model;

import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class DbConfig {
    final static String DriveName="org.sqlite.JDBC";
    public Connection dbConfig() throws SQLException {
        //数据库连接
        Connection connection = null;
        try {
            Class.forName(DriveName);

            connection = DriverManager.getConnection("jdbc:sqlite:ProteinDatabase.db");

//            if (connection != null) {
//                System.out.println("数据库连接成功！");
//            } else {
//                System.out.println("数据库连接失败… ");
//            }

        } catch (ClassNotFoundException | SQLException e) {
            throw new RuntimeException(e);
        }
        return connection;
    }

    public boolean dbConfigThen(){
        Connection connection = null;
        try {
            Class.forName(DriveName);

            connection = DriverManager.getConnection("jdbc:sqlite:ProteinDatabase.db");

            if(connection!=null){
                return true;
            }
            else return false;

        } catch (ClassNotFoundException | SQLException e) {
            throw new RuntimeException(e);
        }
    }
}
