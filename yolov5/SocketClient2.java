
import java.io.*;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.SocketAddress;
import java.nio.file.Files;

public class SocketClient2 {

    public static void main(String[] args) {

        final String hostName = "127.0.0.1";
        final int port = 9999;

        try {
            Socket socket = new Socket();
            SocketAddress address = new InetSocketAddress(hostName, port);
            socket.connect(address);

            OutputStream outputStream = socket.getOutputStream();
            InputStream inputStream = socket.getInputStream();
            BufferedOutputStream bos = new BufferedOutputStream(outputStream);
            BufferedInputStream bis = new BufferedInputStream(inputStream);

            File file = new File("/Users/eunbin/Desktop/GDSC/images/14.jpg");

            byte[] byteFile = Files.readAllBytes(file.toPath());
            System.out.println("전송된 바이트 수 " + byteFile.length);
            bos.write(byteFile);
            bos.flush();
            bos.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
