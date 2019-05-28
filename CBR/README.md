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
## Getting started

### Set up libs
Download mycbr-rest and mycbr-sdk and structure them in folders as instructed above. Then, assuming you're in `CBR/libs/mycbr-rest` and have maven installed, run:
```sh
# Build sdk
cd ../mycbr-sdk
mvn clean install
# Build rest
cd ../mycbr-rest
mvn install:install-file -Dfile=../mycbr-sdk/target/myCBR-3.3-SNAPSHOT.jar -DpomFile=../mycbr-sdk/pom.xml -DlocalRepositoryPath=lib/no/ntnu/mycbr/mycbr-sdk/
mvn clean install
```

To run a project (still in `CBR/libs/mycbr-rest`):
```sh
# Operations are only persistent in memory
java -DMYCBR.PROJECT.FILE=/path/to/project.prj -jar ./target/mycbr-rest-1.0-SNAPSHOT.jar
# With the save option, all operations are saved to the .prj file
java -DMYCBR.PROJECT.FILE=/path/to/project.prj -Dsave=true -jar ./target/mycbr-rest-1.0-SNAPSHOT.jar
```

### Rebuilding after changes to REST
If you've made changes in the source code (bugfix, improvements, etc.) the project needs to be rebuilt before the changes start applying.
Assuming you're in `CBR/libs/mycbr-rest`:
```sh
mvn install:install-file -Dfile=../mycbr-sdk/target/myCBR-3.3-SNAPSHOT.jar -DpomFile=../mycbr-sdk/pom.xml -DlocalRepositoryPath=lib/no/ntnu/mycbr/mycbr-sdk/
mvn clean install 
```
Or run `rebuild_rest.sh` 