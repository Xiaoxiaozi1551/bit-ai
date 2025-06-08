package model;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;

public class Update {
    Connection connection=new DbConfig().dbConfig();

    public Update() throws SQLException {
    }

    public boolean updateData(String TableName,String Column,String newData,String addColumn,String condition) throws SQLException {
        String sql="update "+TableName+" set "+Column+"=? where "+addColumn+"=?";
        PreparedStatement pst=connection.prepareStatement(sql);
        pst.setObject(1,newData);
        pst.setObject(2,condition);
        int i=pst.executeUpdate();
        return i>1;
    }

}
