using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace kMeans
{    public class KMeansResult
    {
        /// <summary>
        /// Label provided by the user. 
        /// </summary>
        public string Label {  get; set; }

        /// <summary>
        /// ID of the KMeans determined cluster for this KMeansResult. ID values monotonically increase from a base of 1. 
        /// </summary>
        public uint PredictedLabel { get; set; }

        /// <summary>
        /// A VBuffer of floats which represent the distances this KMeansResult to the centroid of each cluster. This VBuffer will have one result for each cluster. 
        /// The PredicitedLabel of this KMeansResult can index into the Score to determine the distance to the assigned cluster. Because PredicitedLabel is a 1 based collection and VBuffer is a 0 based collection users must subtract 1 from the PredicitedLabel to perform this index. For example Score.GetValues()[PredictedLabel-1].
        /// </summary>
        public VBuffer<Single> Score { get; set; }
        
        /// <summary>
        /// Features provided by the user. 
        /// </summary>
        public VBuffer<Single> Features{ get; set; }
    }
}
