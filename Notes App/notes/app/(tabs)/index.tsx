import { ScrollView, StyleSheet, View, TouchableOpacity, RefreshControl } from 'react-native';
import { useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { IconSymbol } from '@/components/ui/icon-symbol';
import { Colors } from '@/constants/theme';
import { getReports, saveReport, ClinicalReport } from '@/utils/reports';

const defaultClinicalReportContent = `======================================================================
CLINICAL EXAMINATION REPORT
======================================================================
Report Generated: 2025-11-15 11:37:49
Audio Recording: exam_audio_20251115_113704.wav
======================================================================

**Clinical Examination Report**

---

**Patient Information:**

- **Name:** [Patient Name]
- **Age:** [Patient Age]
- **Date of Examination:** [Examination Date]
- **Examining Physician:** [Your Name]

---

**1. Patient Vital Signs**

- **Heart Rate:** 85 bpm (normal resting heart rate within the range of 60-100 bpm)
- **Blood Oxygen Level:** 95% (normal range is typically 95-100%)
- **Height:** 5'10"

---

**2. Physical Examination Findings**

Upon visual observation during the examination, the following was noted regarding the environment and patient appearance:

- The patient was primarily observed in a modern indoor setting characterized by large windows, wooden elements, and well-lit spaces suggestive of contemporary architectural design. This environment included areas resembling offices, study spaces, or public indoor areas.
  
- The patient was frequently depicted looking downwards, often appearing focused or engaged in a task. This repeated behavior was observed in various settings suggesting a workspace or study area.
  
- On multiple occasions, the patient displayed a neutral and contemplative expression, suggesting engagement with a task or introspection.
  
- At some points, the patient was observed resting their chin on their hand, which can indicate contemplation or concentration.
  
- The patient's immediate environment was noted to include features such as industrial design elements, wooden paneling, and visible structural elements in the ceiling.

---

**3. Patient Communication**

The audio transcript from the examination captured the patient stating, "I quit." This brief communication may indicate a significant decision or change in the patient's current activity or situation.

---

**4. Clinical Assessment**

Based on the visual observations and patient communication, the patient appears to be in a stable physical condition with vital signs within normal limits. The patient's demeanor, expressions, and environment suggest a professional or academic setting where they may be experiencing stress, focus, or decision-making reflective in their communication of potentially leaving a current position or task.

---

**5. Recommendations**

- **Mental Health Evaluation:** Given the communication of "I quit," it may be beneficial to assess the patient's psychological well-being. Explore potential sources of stress, anxiety, or dissatisfaction in the patient's environment or activities.

- **Occupational or Lifestyle Guidance:** Discuss potential changes the patient may be contemplating and provide resources or counseling as needed to support decision-making or transitions in professional or academic life.

- **Follow-Up Appointment:** Schedule a follow-up appointment to monitor any developments in the patient's condition or decisions, and to provide ongoing support as required.

- **Encourage Open Communication:** Provide the patient with a platform to discuss feelings and concerns honestly in future consultations, ensuring a supportive environment for addressing any issues.

---

**Report Prepared by:**

[Your Full Name, Credentials]

[Your Contact Information]

[Date of Report]

======================================================================
END OF REPORT
======================================================================`;

export default function HomeScreen() {
  const router = useRouter();
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';
  const [reports, setReports] = useState<ClinicalReport[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadReports();
    // Auto-save the default report if no reports exist
    initializeDefaultReport();
  }, []);

  async function loadReports() {
    try {
      const loadedReports = await getReports();
      setReports(loadedReports.sort((a, b) => b.createdAt - a.createdAt));
    } catch (error) {
      console.error('Error loading reports:', error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  async function initializeDefaultReport() {
    try {
      const existingReports = await getReports();
      if (existingReports.length === 0) {
        // Save the default report
        await saveReport({
          date: '2025-11-15 11:37:49',
          heartRate: 85,
          bloodAlcohol: 95,
          bloodPressure: '120/80',
          height: '5\'10"',
          content: defaultClinicalReportContent,
        });
        loadReports();
      }
    } catch (error) {
      console.error('Error initializing default report:', error);
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

      {reports.length === 0 ? (
        <ThemedView style={styles.emptyContainer}>
          <IconSymbol name="doc.text" size={64} color={isDark ? '#666' : '#999'} />
          <ThemedText style={styles.emptyText}>No reports yet</ThemedText>
        </ThemedView>
      ) : (
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
      )}
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
});
