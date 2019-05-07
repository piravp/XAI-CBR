package no.ntnu.mycbr.rest;

import no.ntnu.mycbr.core.Project;
import no.ntnu.mycbr.CBREngine;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import springfox.documentation.swagger2.annotations.EnableSwagger2;

/**
 * Hello world!
 *
 */
@SpringBootApplication
@EnableSwagger2
public class App {

    public static Project getProject() {
        return project;
    }

    private static CBREngine engine;
    private static Project project;

    public static void main(String[] args) {

        engine = new CBREngine();
        String userDefinedProjectFile = System.getProperty("MYCBR.PROJECT.FILE");
        if(userDefinedProjectFile != null && userDefinedProjectFile.length() > 0)
            project = engine.createProjectFromPRJ(userDefinedProjectFile);
        else{
            project = engine.createemptyCBRProject();
        }

        SpringApplication.run(App.class, args);
    }

}
