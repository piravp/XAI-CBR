package no.ntnu.mycbr.rest.utils;

import no.ntnu.mycbr.rest.Case;
import no.ntnu.mycbr.rest.Query;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class RESTCBRUtils {
    public static List<LinkedHashMap<String, String>> getFullResult(Query query, String concept) {
        LinkedHashMap<String, Double> results = query.getSimilarCases();
        List<LinkedHashMap<String, String>> cases = new ArrayList<>();

        for (Map.Entry<String, Double> entry : results.entrySet()) {
            String entryCaseID = entry.getKey();
            double similarity = entry.getValue();
            Case caze = new Case(concept, entryCaseID, similarity);
            cases.add(caze.getCase());
        }

        return cases;
    }
}
