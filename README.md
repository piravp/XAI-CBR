# Explainable AI using CBR
Explaining DNN using CBR as explanation engine.

## Setup
#### Python packages
Install packages in `requirements.txt`. The easiest is to use `pip`.

<!-- Installing myCBR etc. -->
#### File Structure
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


#### Set up libs
Download [mycbr-rest](https://github.com/ntnu-ai-lab/mycbr-rest) and [mycbr-sdk](https://github.com/ntnu-ai-lab/mycbr-sdk) and structure them in folders as instructed above. Then, assuming you're in `CBR/libs/mycbr-rest` and have maven installed, run:
```sh
# Build sdk
cd ../mycbr-sdk
mvn clean install
# Build rest
cd ../mycbr-rest
mvn install:install-file -Dfile=../mycbr-sdk/target/myCBR-3.3-SNAPSHOT.jar -DpomFile=../mycbr-sdk/pom.xml -DlocalRepositoryPath=lib/no/ntnu/mycbr/mycbr-sdk/
mvn clean install
```

#### myCBR workbench
Not required, but helpful to view the project in a GUI. 
Can be found here: http://mycbr-project.org/download.html

## Run
Run these command to reproduce results in thesis. Note that these operation start up a myCBR server instance every time, so expect this to take some time.


### Populate CB
Populate case-base with cases. Default is set to 5.
```
python main.py fill_final
```

### Retrieve
Retrieve most similar case from the case-base.
```sh
python main.py retrieve
```

### Retain
Retaining a test-case can be done by setting storage in the retrieve step to `true`. Default is `false`.


---
To run server standalone (still in `CBR/libs/mycbr-rest`):
```sh
# Operations are only persistent in memory
java -DMYCBR.PROJECT.FILE=/path/to/project.prj -jar ./target/mycbr-rest-1.0-SNAPSHOT.jar
# With the save option, all operations are saved to the .prj file
java -DMYCBR.PROJECT.FILE=/path/to/project.prj -Dsave=true -jar ./target/mycbr-rest-1.0-SNAPSHOT.jar
```