# Build rest
mvn install:install-file -Dfile=../mycbr-sdk/target/myCBR-3.3-SNAPSHOT.jar -DpomFile=../mycbr-sdk/pom.xml -DlocalRepositoryPath=lib/no/ntnu/mycbr/mycbr-sdk/
mvn clean install