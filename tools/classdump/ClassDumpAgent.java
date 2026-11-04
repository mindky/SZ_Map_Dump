import java.io.File;
import java.io.FileOutputStream;
import java.lang.instrument.ClassFileTransformer;
import java.lang.instrument.Instrumentation;
import java.security.ProtectionDomain;

/**
 * Dumps the decrypted bytecode of every non-JDK class as it is loaded.
 *
 * The game's custom JVM decrypts classes.jar entries before defining them, so
 * a ClassFileTransformer sees the plaintext class bytes. Run via
 * JAVA_TOOL_OPTIONS=-javaagent:classdump.jar=<output dir> (the JVM picks the
 * variable up automatically on launch).
 */
public final class ClassDumpAgent {
    private static File outDir;

    public static void premain(String args, Instrumentation inst) {
        String dir = (args == null || args.trim().isEmpty()) ? "classdump" : args.trim();
        outDir = new File(dir);
        outDir.mkdirs();

        inst.addTransformer(new ClassFileTransformer() {
            @Override
            public byte[] transform(
                    ClassLoader loader,
                    String name,
                    Class<?> clazz,
                    ProtectionDomain domain,
                    byte[] bytes) {
                if (name == null || bytes == null || isJdkClass(name)) {
                    return null;
                }
                try {
                    File target = new File(outDir, name + ".class");
                    File parent = target.getParentFile();
                    if (parent != null) {
                        parent.mkdirs();
                    }
                    try (FileOutputStream fos = new FileOutputStream(target)) {
                        fos.write(bytes);
                    }
                } catch (Exception ignored) {
                    // never break class loading because of a dump failure
                }
                return null; // do not modify the class
            }
        });

        System.out.println("[ClassDumpAgent] dumping non-JDK classes to " + outDir.getAbsolutePath());
    }

    private static boolean isJdkClass(String name) {
        return name.startsWith("java/")
                || name.startsWith("javax/")
                || name.startsWith("jdk/")
                || name.startsWith("sun/")
                || name.startsWith("com/sun/")
                || name.startsWith("jdk/");
    }
}
