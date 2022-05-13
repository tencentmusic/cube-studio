/*
* Copyright 2010-2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License").
* You may not use this file except in compliance with the License.
* A copy of the License is located at
*
*  http://aws.amazon.com/apache2.0
*
* or in the "license" file accompanying this file. This file is distributed
* on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
* express or implied. See the License for the specific language governing
* permissions and limitations under the License.
*/

#pragma once

#include <aws/transfer/Transfer_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSSet.h>
#include <aws/core/utils/memory/stl/AWSMap.h>
#include <aws/core/utils/UUID.h>
#include <aws/core/client/AWSError.h>
#include <aws/core/client/AsyncCallerContext.h>
#include <aws/s3/S3Errors.h>
#include <iostream>
#include <atomic>
#include <mutex>
#include <condition_variable>

namespace Aws
{
    namespace Utils
    {
        template < typename T > class Array;
    }

    namespace Transfer
    {
        class TransferHandle;

        typedef std::function<Aws::IOStream*(void)> CreateDownloadStreamCallback;

        static const char CLASS_TAG[] = "TransferManager";

        struct DownloadConfiguration
        {
            DownloadConfiguration() :
                versionId("")
            {}

            Aws::String versionId;

            // TBI: controls for in-memory parts vs. resumable file-based parts with state serialization to/from file
        };

        class PartState
        {
            public:
                PartState();
                PartState(int partId, size_t bestProgressInBytes, size_t sizeInBytes, bool lastPart = false);

                int GetPartId() const { return m_partId; }

                size_t GetBestProgressInBytes() const { return m_bestProgressInBytes; }
                void SetBestProgressInBytes(size_t progressInBytes) { m_bestProgressInBytes = progressInBytes; }

                size_t GetSizeInBytes() const { return m_sizeInBytes; }
                void SetSizeInBytes(size_t sizeInBytes) { m_sizeInBytes = sizeInBytes; }

                void Reset();

                void OnDataTransferred(long long amount, const std::shared_ptr<TransferHandle> &transferHandle);

                void SetETag(const Aws::String& eTag) { m_eTag = eTag; }
                const Aws::String& GetETag() const { return m_eTag; }

                Aws::IOStream *GetDownloadPartStream() const { return m_downloadPartStream; }
                void SetDownloadPartStream(Aws::IOStream *downloadPartStream) { m_downloadPartStream = downloadPartStream; }

                unsigned char* GetDownloadBuffer() const { return m_downloadBuffer; }
                void SetDownloadBuffer(unsigned char* downloadBuffer) { m_downloadBuffer = downloadBuffer; }

                void SetRangeBegin(size_t rangeBegin) { m_rangeBegin = rangeBegin; }
                size_t GetRangeBegin() const { return m_rangeBegin; }

                bool IsLastPart() { return m_lastPart; }
                void SetLastPart() { m_lastPart = true; }

            private:

                int m_partId;

                Aws::String m_eTag;
                size_t m_currentProgressInBytes;
                size_t m_bestProgressInBytes;
                size_t m_sizeInBytes;
                size_t m_rangeBegin;

                std::atomic<Aws::IOStream *> m_downloadPartStream;
                std::atomic<unsigned char*> m_downloadBuffer;
                bool m_lastPart;
        };

        using PartPointer = std::shared_ptr< PartState >;
        using PartStateMap = Aws::Map< int, PartPointer >;

        enum class TransferStatus
        {
            //this value is only used for directory synchronization
            EXACT_OBJECT_ALREADY_EXISTS,    
            //Operation is still queued and has not begun processing        
            NOT_STARTED,
            //Operation is now running
            IN_PROGRESS,
            //Operation was canceled. A Canceled operation can still be retried
            CANCELED,
            //Operation failed, A failed operaton can still be retried.
            FAILED,
            //Operation was successful
            COMPLETED,
            //Operation either failed or was canceled and a user deleted the multi-part upload from S3.
            ABORTED
        };

        enum class TransferDirection
        {
            UPLOAD,
            DOWNLOAD
        };

        /**
         * This is the interface for interacting with an in-process transfer. All operations from TransferManager return an instance of this class.
         * In addition to the status of the transfer and details about what operation is being performed, this class also has the Cancel() operation which is
         * used to cancel a transfer, and WaitUntilCompleted() which will cause the calling thread to block until the transfer is finished.
         *
         * In the context that by the time you are using this class, it is thread safe.
         */
        class AWS_TRANSFER_API TransferHandle
        {
        public:
            /**
             * Initialize with required information for an UPLOAD
             */
            TransferHandle(const Aws::String& bucketName, const Aws::String& keyName, uint64_t totalSize, const Aws::String& targetFilePath = "");

            /**
             * Initialize with required information for a DOWNLOAD
             */
            TransferHandle(const Aws::String& bucketName, const Aws::String& keyName, const Aws::String& targetFilePath = "");

            /**
             * Alternate DOWNLOAD constructor
             */
            TransferHandle(const Aws::String& bucketName, const Aws::String& keyName, CreateDownloadStreamCallback createDownloadStreamFn, const Aws::String& targetFilePath = "");

            /**
             * Alternate DOWNLOAD constructor
             */
            TransferHandle(const Aws::String& bucketName, const Aws::String& keyName, 
                const uint64_t fileOffset, const uint64_t downloadBytes, 
                CreateDownloadStreamCallback createDownloadStreamFn, const Aws::String& targetFilePath = "");


            ~TransferHandle();

            /**
             * Whether or not this transfer is being performed using parallel parts via a multi-part s3 api.
             */
            inline bool IsMultipart() const { return m_isMultipart.load(); }
            /**
            * Whether or not this transfer is being performed using parallel parts via a multi-part s3 api.
            */
            inline void SetIsMultipart(bool value) { m_isMultipart.store(value); }
            /**
            * If this is a multi-part transfer, this is the ID of it. e.g. UploadId for UploadPart
            */
            inline const Aws::String GetMultiPartId() const { std::lock_guard<std::mutex> locker(m_getterSetterLock); return m_multipartId; }
            /**
            * If this is a multi-part transfer, this is the ID of it. e.g. UploadId for UploadPart
            */
            inline void SetMultipartId(const Aws::String& value) { std::lock_guard<std::mutex> locker(m_getterSetterLock); m_multipartId = value; }
            /**
             * Returns a copy of the completed parts, in the structure of <partId, ETag>. Used for all transfers.
             */
            PartStateMap GetCompletedParts() const;
            /**
             * Set a pending part to completed along with its etag. Used fore all transfers.
             */
            void ChangePartToCompleted(const PartPointer& partState, const Aws::String &eTag);
            /**
             * Returns a copy of the pending parts. Used for all transfers.
             */
            PartStateMap GetPendingParts() const;
            /**
            * Returns true or false if there are currently any pending parts.
            */
            bool HasPendingParts() const;
            /**
             * Set a part to pending. Used for all transfers.
             */
            void AddPendingPart(const PartPointer& partState);
            /**
             * Returns a copy of the queued parts. Used for all transfers.
             */
            PartStateMap GetQueuedParts() const;
            /**
            * Returns true or false if there are currently any queued parts.
            */
            bool HasQueuedParts() const;
            /**
             * Set a part to queued. Used for all transfers.
             */
            void AddQueuedPart(const PartPointer& partState);
            /**
             * Returns a copy of the failed parts. Used for all transfers.
             */
            PartStateMap GetFailedParts() const;
            /**
            * Returns true or false if there are currently any failed parts.
            */
            bool HasFailedParts() const;
            /**
             * Set a pending part to failed. Used for all transfers.
             */
            void ChangePartToFailed(const PartPointer& partState);
            /**
             * Get the parts transactionally, mostly for internal purposes.
             */
            void GetAllPartsTransactional(PartStateMap& queuedParts, PartStateMap& pendingParts,
                    PartStateMap& failedParts, PartStateMap& completedParts);
            /**
             * Returns true or false if any parts have been created for this transfer
             */
            bool HasParts() const;
            /**
             * Returns false if Cancel has been called. Largely for internal use.
             */
            bool ShouldContinue() const;
            /**
             * Cancel the transfer. This will happen asynchronously, so if you need to wait for it to be canceled, either handle the callbacks,
             *  or call WaitUntilFinished.
             */
            void Cancel();

            /**
             * Reset the cancellation status for a retry. This will be done automatically by Transfermanager.
             */
            void Restart();
            /**
             * Total bytes transferred successfully on this transfer operation.
             * We implement transfer progress with two invariants:
             *  (1) Never lock; given a callback that can happen hundreds of times a second or more on a solid connection, it isn't acceptable to lock each time
             *  (2) Never go backwards, in spite of part upload/download failures.  Negative progress (canceling a highly concurrent transfer can
             *      lead to an enormous step backwards if many parts are aborted at once) is a confusing and undesirable user experience.
             * In this sense, progress represents a high-water mark, and in the presence of heavy failures or cancellation, it may appear to pause until the 
             * necessary retries exceed the previous high-water mark.
             */
            inline uint64_t GetBytesTransferred() const { return m_bytesTransferred.load(); }
            /**
            * Total bytes transferred successfully on this transfer operation.
            */
            void UpdateBytesTransferred(uint64_t amount) { m_bytesTransferred.fetch_add(amount); }

            /**
             * The offset from which to start downloading
             */
            inline uint64_t GetBytesOffset() const { return m_offset; }
            /**
             * The calculated total size of the object being transferred.
             */
            inline uint64_t GetBytesTotalSize() const { return m_bytesTotalSize.load(); }
            /**
             * Sets the total size of the object being transferred.
             */
            inline void SetBytesTotalSize(uint64_t value) { m_bytesTotalSize.store(value); }

            /**
             * Bucket portion of the object location in Amazon S3.
             */
            inline const Aws::String& GetBucketName() const { return m_bucket; }
            /**
             * Key of the object location in Amazon S3.
             */
            inline const Aws::String& GetKey() const { return m_key; }
            /**
             * If known, this is the location of the local file being uploaded from, or downloaded to. If you use the stream api however, this will
             * always be blank.
             */
            inline const Aws::String& GetTargetFilePath() const { return m_fileName; }

            /**
             * (Download only) version id of the object to retrieve; if not specified in constructor, then latest is used
            */
            const Aws::String GetVersionId() const { std::lock_guard<std::mutex> locker(m_getterSetterLock); return m_versionId; }
            void SetVersionId(const Aws::String& versionId) { std::lock_guard<std::mutex> locker(m_getterSetterLock); m_versionId = versionId; }

            /**
             * Upload or Download?
             */
            inline TransferDirection GetTransferDirection() const { return m_direction; }
            /**
             * Content type of the object being transferred
             */
            inline const Aws::String GetContentType() const { std::lock_guard<std::mutex> locker(m_getterSetterLock); return m_contentType; }
            /**
             * Content type of the object being transferred
             */
            inline void SetContentType(const Aws::String& value) { std::lock_guard<std::mutex> locker(m_getterSetterLock); m_contentType = value; } 
            /**
             * In case of an upload, this is the metadata that was placed on the object when it was uploaded.
             * In the case of a download, this is the object metadata from the GetObject operation.
             */
            inline const Aws::Map<Aws::String, Aws::String> GetMetadata() const { std::lock_guard<std::mutex> locker(m_getterSetterLock); return m_metadata; }
            /**
            * In case of an upload, this is the metadata that was placed on the object when it was uploaded.
            * In the case of a download, this is the object metadata from the GetObject operation.
            */
            inline void SetMetadata(const Aws::Map<Aws::String, Aws::String>& value) { std::lock_guard<std::mutex> locker(m_getterSetterLock); m_metadata = value; }

            /**
             * Add a new entry to or update an existed entry of m_metadata, useful when users want to get ETag directly from metadata.
             */
            inline void AddMetadataEntry(const Aws::String& key, const Aws::String& value) { std::lock_guard<std::mutex> locker(m_getterSetterLock); m_metadata[key] = value; }

            /**
             * Arbitrary user context that can be accessed from the callbacks
             */
            inline void SetContext(const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) { std::lock_guard<std::mutex> locker(m_getterSetterLock); m_context = context; }

            /**
             * Returns arbitrary user context or nullptr if it's not set.
             */
            inline std::shared_ptr<const Aws::Client::AsyncCallerContext> GetContext() const { std::lock_guard<std::mutex> locker(m_getterSetterLock); return m_context; }

            /**
             * The current status of the operation
             */
            TransferStatus GetStatus() const;
            /**
            * The current status of the operation
            */
            void UpdateStatus(TransferStatus value);

            /**
             * The last error that was encountered by the transfer. You can handle each error individually via the errorCallback callback function
             * in the TransferConfiguration.
             */
            inline const Aws::Client::AWSError<Aws::S3::S3Errors> GetLastError() const { std::lock_guard<std::mutex> locker(m_getterSetterLock); return m_lastError; }
            /**
             * The last error that was encountered by the transfer. You can handle each error individually via the errorCallback callback function
             * in the TransferConfiguration.
             */
            inline void SetError(const Aws::Client::AWSError<Aws::S3::S3Errors>& error) { std::lock_guard<std::mutex> locker(m_getterSetterLock); m_lastError = error; }
            /**
             * Blocks the calling thread until the operation has finished. This function does not busy wait. It is safe for your CPU.
             */
            void WaitUntilFinished() const;      

            const CreateDownloadStreamCallback& GetCreateDownloadStreamFunction() const { return m_createDownloadStreamFn; }

            void WritePartToDownloadStream(Aws::IOStream* partStream, std::size_t writeOffset);

            void ApplyDownloadConfiguration(const DownloadConfiguration& downloadConfig);

            bool LockForCompletion() 
            {
                bool expected = false;
                return m_lastPart.compare_exchange_strong(expected, true/*desired*/);
            }

            /*
             * Returns a unique identifier tied to this particular transfer handle.
             */
            Aws::String GetId() const;

        private:

            void CleanupDownloadStream();

            std::atomic<bool> m_isMultipart;
            Aws::String m_multipartId;
            TransferDirection m_direction;
            PartStateMap m_completedParts;
            PartStateMap m_pendingParts;
            PartStateMap m_queuedParts;
            PartStateMap m_failedParts;
            std::atomic<uint64_t> m_bytesTransferred;
            std::atomic<bool> m_lastPart;
            std::atomic<uint64_t> m_bytesTotalSize;
            uint64_t m_offset;
            Aws::String m_bucket;
            Aws::String m_key;
            Aws::String m_fileName;
            Aws::String m_contentType;
            Aws::String m_versionId;
            Aws::Map<Aws::String, Aws::String> m_metadata;
            TransferStatus m_status;
            Aws::Client::AWSError<Aws::S3::S3Errors> m_lastError;
            std::atomic<bool> m_cancel;
            std::shared_ptr<const Aws::Client::AsyncCallerContext> m_context;
            const Utils::UUID m_handleId;

            CreateDownloadStreamCallback m_createDownloadStreamFn;
            Aws::IOStream* m_downloadStream;

            mutable std::mutex m_downloadStreamLock;
            mutable std::mutex m_partsLock;
            mutable std::mutex m_statusLock;
            mutable std::condition_variable m_waitUntilFinishedSignal;
            mutable std::mutex m_getterSetterLock;
        };

        AWS_TRANSFER_API Aws::OStream& operator << (Aws::OStream& s, TransferStatus status);
    }
}
