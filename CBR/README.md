### File structure
A guide on how the files should be structured.
```
CBR
 |__ libs
        |__ mycbr-sdk
                |__pom.xml
                |__src
                |__target
                    |__myCBR-x.x-SNAPSHOT.jar
        |__ mycbr-rest
                |__pom.xml
                |__src
                |__target
                    |__mycbr-rest-x.x-SNAPSHOT.jar
                |__lib/no/ntnu/mycbr/mycbr-sdk/
```

### Rebuilding after changes to REST
After changes has been made to the source code of the rest API, the projects needs to be rebuilt for the changes to start applying. 
Assuming you're in `CBR/libs/mycbr-rest`:
```sh
mvn install:install-file -Dfile=../mycbr-sdk/target/myCBR-3.3-SNAPSHOT.jar -DpomFile=../mycbr-sdk/pom.xml -DlocalRepositoryPath=lib/no/ntnu/mycbr/mycbr-sdk/
mvn clean install 
```