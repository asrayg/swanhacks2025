import { ScrollView, StyleSheet, View, TouchableOpacity, RefreshControl } from 'react-native';
import { useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { IconSymbol } from '@/components/ui/icon-symbol';
import { Colors } from '@/constants/theme';
import { getReports, ClinicalReport } from '@/utils/reports';

export default function HomeScreen() {
  const router = useRouter();
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';
  const [reports, setReports] = useState<ClinicalReport[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadReports();
  }, []);

  async function loadReports() {
    try {
      setError(null);
      console.log('Loading reports...');
      const loadedReports = await getReports();
      console.log('Loaded reports:', loadedReports.length);
      setReports(loadedReports.sort((a, b) => b.createdAt - a.createdAt));
    } catch (error: any) {
      console.error('Error loading reports:', error);
      const errorMessage = error?.message || 'Failed to load reports. Please check your connection and try again.';
      setError(errorMessage);
      // Set empty array on error so user can see the empty state
      setReports([]);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  function handleRefresh() {
    setRefreshing(true);
    loadReports();
  }

  function handleReportPress(report: ClinicalReport) {
    router.push(`/(tabs)/report/${report.id}`);
  }

  if (loading) {
    return (
      <ThemedView style={styles.container}>
        <ThemedText>Loading...</ThemedText>
      </ThemedView>
    );
  }

  return (
    <ScrollView 
      style={[styles.container, { backgroundColor: isDark ? Colors.dark.background : Colors.light.background }]}
      contentContainerStyle={styles.contentContainer}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={handleRefresh} />
      }
    >
      <ThemedView style={styles.header}>
        <ThemedText type="title" style={styles.title}>Clinical Reports</ThemedText>
        <ThemedText style={styles.subtitle}>{reports.length} {reports.length === 1 ? 'Report' : 'Reports'}</ThemedText>
      </ThemedView>

      {error && (
        <ThemedView style={[
          styles.errorContainer,
          { backgroundColor: isDark ? '#2C1810' : '#FFF5F5' }
        ]}>
          <IconSymbol name="exclamationmark.triangle.fill" size={48} color="#E74C3C" />
          <ThemedText style={styles.errorText}>{error}</ThemedText>
          <TouchableOpacity 
            onPress={loadReports}
            style={styles.retryButton}
          >
            <ThemedText style={styles.retryButtonText}>Retry</ThemedText>
          </TouchableOpacity>
        </ThemedView>
      )}

      {!error && reports.length === 0 ? (
        <ThemedView style={styles.emptyContainer}>
          <IconSymbol name="doc.text" size={64} color={isDark ? '#666' : '#999'} />
          <ThemedText style={styles.emptyText}>No reports yet</ThemedText>
        </ThemedView>
      ) : !error ? (
        <View style={styles.reportsList}>
          {reports.map((report) => (
            <TouchableOpacity
              key={report.id}
              onPress={() => handleReportPress(report)}
              style={[
                styles.reportCard,
                {
                  backgroundColor: isDark ? '#1E1E1E' : '#FFFFFF',
                  shadowColor: isDark ? '#000' : '#000',
                }
              ]}
            >
              <View style={styles.reportCardHeader}>
                <View style={styles.reportCardIcon}>
                  <IconSymbol name="doc.text.fill" size={24} color={Colors.light.tint} />
                </View>
                <View style={styles.reportCardInfo}>
                  <ThemedText type="defaultSemiBold" style={styles.reportCardTitle}>
                    Clinical Report
                  </ThemedText>
                  <ThemedText style={styles.reportCardDate}>{report.date}</ThemedText>
                </View>
                <IconSymbol name="chevron.right" size={20} color={isDark ? '#666' : '#999'} />
              </View>
              <View style={styles.reportCardVitals}>
                <View style={styles.reportCardVital}>
                  <IconSymbol name="heart.fill" size={16} color="#E74C3C" />
                  <ThemedText style={styles.reportCardVitalText}>{report.heartRate} bpm</ThemedText>
                </View>
                <View style={styles.reportCardVital}>
                  <IconSymbol name="testtube.2" size={16} color="#F39C12" />
                  <ThemedText style={styles.reportCardVitalText}>{report.bloodAlcohol}%</ThemedText>
                </View>
                <View style={styles.reportCardVital}>
                  <IconSymbol name="waveform.path.ecg" size={16} color="#9B59B6" />
                  <ThemedText style={styles.reportCardVitalText}>{report.bloodPressure || '120/80'} mm</ThemedText>
                </View>
                <View style={styles.reportCardVital}>
                  <IconSymbol name="ruler.fill" size={16} color="#3498DB" />
                  <ThemedText style={styles.reportCardVitalText}>{report.height}</ThemedText>
                </View>
              </View>
            </TouchableOpacity>
          ))}
        </View>
      ) : null}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  contentContainer: {
    padding: 20,
    paddingBottom: 40,
  },
  header: {
    marginBottom: 24,
    alignItems: 'center',
  },
  title: {
    marginTop: 55,
    marginBottom: 8,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    opacity: 0.7,
    textAlign: 'center',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyText: {
    marginTop: 16,
    fontSize: 16,
    opacity: 0.6,
  },
  reportsList: {
    gap: 16,
  },
  reportCard: {
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.08)',
  },
  reportCardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  reportCardIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(10, 126, 164, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  reportCardInfo: {
    flex: 1,
  },
  reportCardTitle: {
    fontSize: 16,
    marginBottom: 4,
  },
  reportCardDate: {
    fontSize: 14,
    opacity: 0.6,
  },
  reportCardVitals: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: 'rgba(0,0,0,0.05)',
  },
  reportCardVital: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginRight: 8,
  },
  reportCardVitalText: {
    fontSize: 12,
    opacity: 0.8,
  },
  errorContainer: {
    marginTop: 20,
    padding: 20,
    borderRadius: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#E74C3C',
  },
  errorText: {
    marginTop: 12,
    marginBottom: 16,
    fontSize: 14,
    color: '#E74C3C',
    textAlign: 'center',
  },
  retryButton: {
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
    backgroundColor: '#E74C3C',
  },
  retryButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
});
