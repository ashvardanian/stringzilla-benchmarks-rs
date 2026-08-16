package com.stringzilla;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.apache.lucene.analysis.core.KeywordAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.FuzzyQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TotalHitCountCollector;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.search.AutomatonQuery;
import org.apache.lucene.util.automaton.LevenshteinAutomata;

public final class LevenshteinIndexBenchmark {
    private static List<String> loadAscii(Path path) throws Exception {
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        for (String line : lines)
            for (int index = 0; index != line.length(); ++index)
                if (line.charAt(index) > 0x7f)
                    throw new IllegalArgumentException("non-ASCII input would compare different semantics: " + path);
        return lines;
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 4) {
            System.err.println("usage: LevenshteinIndexBenchmark DICTIONARY QUERIES MAX_DISTANCE fuzzy|automaton");
            System.exit(2);
        }
        List<String> dictionary = loadAscii(Path.of(args[0]));
        List<String> queries = loadAscii(Path.of(args[1]));
        int maxDistance = Integer.parseInt(args[2]);
        boolean exactAutomaton = args[3].equals("automaton");
        if (!exactAutomaton && !args[3].equals("fuzzy")) {
            System.err.println("mode must be fuzzy or automaton");
            System.exit(2);
        }
        if (maxDistance < 1 || maxDistance > FuzzyQuery.defaultMaxEdits) {
            System.err.printf("MAX_DISTANCE must be between 1 and %d%n", FuzzyQuery.defaultMaxEdits);
            System.exit(2);
        }

        long buildStart = System.nanoTime();
        ByteBuffersDirectory directory = new ByteBuffersDirectory();
        IndexWriterConfig config = new IndexWriterConfig(new KeywordAnalyzer());
        try (IndexWriter writer = new IndexWriter(directory, config)) {
            for (String word : dictionary) {
                Document document = new Document();
                document.add(new StringField("term", word, Field.Store.NO));
                writer.addDocument(document);
            }
            writer.forceMerge(1);
        }
        DirectoryReader reader = DirectoryReader.open(directory);
        IndexSearcher searcher = new IndexSearcher(reader);
        double buildSeconds = (System.nanoTime() - buildStart) * 1e-9;
        System.out.printf("dictionary=%d queries=%d build=%.6fs%n", dictionary.size(), queries.size(), buildSeconds);

        for (int bound = 1; bound <= maxDistance; ++bound) {
            for (int repeat = 0; repeat != 3; ++repeat) {
                long matches = 0;
                long start = System.nanoTime();
                for (String query : queries) {
                    Term term = new Term("term", query);
                    Query fuzzy = exactAutomaton
                            ? new AutomatonQuery(term, new LevenshteinAutomata(query, false).toAutomaton(bound))
                            : new FuzzyQuery(term, bound, 0, Integer.MAX_VALUE, false);
                    TotalHitCountCollector collector = new TotalHitCountCollector();
                    searcher.search(fuzzy, collector);
                    matches += collector.getTotalHits();
                }
                double seconds = (System.nanoTime() - start) * 1e-9;
                System.out.printf("k=%d query=%.6fs matches=%d%n", bound, seconds, matches);
            }
        }
        reader.close();
        directory.close();
    }
}
