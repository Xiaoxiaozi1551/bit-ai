package model;

import java.sql.*;

public class Delete {
    Connection connection=new DbConfig().dbConfig();

    public Delete() throws SQLException {
    }

    public boolean deleteData(String TableName,String Key,String Word) throws SQLException{
        String sql="delete from "+TableName+" where "+Key+" = ?";
        PreparedStatement pst=connection.prepareStatement(sql);

        pst.setObject(1,Word);
        int i=pst.executeUpdate();
        return i > 0;
    }

    public boolean deleteTable(String TableNames) throws SQLException {
        String sql="drop table if exists "+TableNames;
        Statement statement=connection.createStatement();

        try{
            statement.executeUpdate(sql);
            return true;
        } catch (SQLException e) {
            return false;
        }



    }
}
