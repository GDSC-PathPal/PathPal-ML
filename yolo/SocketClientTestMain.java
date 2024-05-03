import java.io.*;
import java.net.Socket;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

public class SocketClientTestMain {

    private final Socket socket;
    private final OutputStream os;
    private final InputStream is;


    public SocketClientTestMain(String hostName, int port) throws IOException {
        this.socket = new Socket(hostName, port);
        this.os = socket.getOutputStream();
        this.is = socket.getInputStream();
    }

    public void inferenceImage(byte[] file) {

        try {
            BufferedOutputStream bos = new BufferedOutputStream(os);
            BufferedInputStream bis = new BufferedInputStream(is);

            // 1. Send File Size
            System.out.println("파일 사이즈 전송:" + file.length);
            DataOutputStream dataOutputStream = new DataOutputStream(bos);
            dataOutputStream.writeInt(file.length);
            bos.flush();

            // 2. receive data
            byte[] tmp = new byte[10000];
            int zz = bis.read(tmp);

            // 3. send image
            System.out.println("이미지 전송 " + file.length);
            bos.write(file);
            bos.flush();

            // 4. receive data
            int read = bis.read(tmp);
            System.out.println("read = " + read);

            String bytesToStr = new String(tmp, 0, read);
            System.out.println(bytesToStr);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {

        String[] filepaths = {
"/home/dnflrha12/PathPal-ML/yolov5/example.jpg",
"/home/dnflrha12/PathPal-ML/yolov5/example2.jpg"
        };

        SocketClientTestMain socketClientTestMain = null;
        try {
            socketClientTestMain = new SocketClientTestMain("127.0.0.1", 9999);
        } catch (IOException e) {
            System.err.println("SocketClient 생성 실패" + e);
        }

        try {

            for(String filepath: filepaths){
                System.out.println("========================");
                File file = new File(filepath);
                byte[] byteFile = Files.readAllBytes(file.toPath());

                socketClientTestMain.inferenceImage(byteFile);
            }
        } catch (Exception e) {
            System.err.println("파일 송수신 및 변환 실패" + e);
        }
    }
}
